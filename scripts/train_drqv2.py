import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import re
from pathlib import Path

import diffusion_reward.rl.drqv2.utils as utils
import hydra
import numpy as np
import torch
import wandb
from diffusion_reward.envs import make_env
from diffusion_reward.models.reward_models import make_rm
from diffusion_reward.rl.drqv2.logger import Logger
from diffusion_reward.rl.drqv2.replay_buffer import (AMPBuffer,
                                                     ReplayBufferStorage,
                                                     ReplayCache,
                                                     make_replay_loader)
from diffusion_reward.rl.drqv2.video import TrainVideoRecorder, VideoRecorder
from dm_env import specs

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        if self.cfg.use_wandb:
            exp_name = cfg.task_name + '_' + cfg.exp_name + '_' + str(cfg.seed) + '_server1'
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            proj_name = "daw" if cfg.domain == 'adroit' else "daw_" + cfg.domain
            wandb.init(project=proj_name,
                       group=group_name,
                       name=exp_name,
                       config=cfg)
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._nstep = cfg.nstep
        self.use_rm = cfg.use_rm
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(), self.cfg.agent)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # reward model
        if self.use_rm:
            self.cfg.reward.obs_shape = self.train_env.observation_spec().shape
            self.cfg.reward.action_shape = self.train_env.action_spec().shape[0]
            self.rm = make_rm(self.cfg.reward).to(self.device)
            self.replay_cache = ReplayCache(device=self.device)
            self.eval_replay_cache = ReplayCache(device=self.device)
            self.pretrain_rm = self.cfg.reward.pretrain_rm

            if self.cfg.reward.rm_model == 'amp':
                self.amp_buffer = AMPBuffer(batch_size=self.cfg.batch_size, cfg=self.cfg)
        else:
            self.rm = None
            self.pretrain_rm = False
        self.agent.rm = self.rm if self.use_rm else None

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        # create envs
        self.train_env = make_env(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = make_env(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader, self.buffer = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers, self.cfg.save_snapshot,
            self._nstep,
            self._discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward, total_sr = 0, 0, 0, 0
        if self.pretrain_rm:
            total_learned_reward = 0.0

        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        while eval_until_episode(episode):
            episode_sr = False

            time_step = self.eval_env.reset()
            if self.pretrain_rm:
                self.eval_replay_cache.add(time_step)
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                if self.pretrain_rm:
                    self.eval_replay_cache.add(time_step)
                self.video_recorder.record(self.eval_env)
                episode_sr = episode_sr or time_step.is_success
                total_reward += time_step.reward
                step += 1

            total_sr += episode_sr
            episode += 1
            
            if self.pretrain_rm:
                learned_return = self.eval_replay_cache.pop(None, self.rm)
                total_learned_reward += learned_return

            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_success_rate', total_sr / episode)
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            if self.pretrain_rm: 
                log('episode_learned_reward', total_learned_reward / episode)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward, episode_sr = 0, 0, False
        time_step = self.train_env.reset()

        if not self.pretrain_rm:
            self.replay_storage.add(time_step)
        else:
            self.replay_cache.add(time_step)

        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')

                # reset env
                time_step = self.train_env.reset()

                if not self.pretrain_rm:
                    self.replay_storage.add(time_step)
                else:
                    learned_return = self.replay_cache.pop(self.replay_storage, self.rm)
                    self.replay_cache.add(time_step)

                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_success_rate', episode_sr)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                        if self.pretrain_rm:
                            log('episode_learned_reward', learned_return)

                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_sr = False
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step) and self.global_step % self.cfg.update_every_steps == 0:
                batch = next(self.replay_iter)
                metrics = dict()

                if self.use_rm and self.global_step % self.rm.expl_update_interval == 0:
                    if self.cfg.reward.rm_model == 'amp':
                        expert_obs = self.amp_buffer.sample()
                        metrics.update(self.rm.update(batch,  expert_obs))   
                    else:          
                        metrics.update(self.rm.update(batch))
                metrics.update(self.agent.update(batch, self.global_step))

                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            episode_sr = episode_sr or time_step.is_success
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward

            if not self.pretrain_rm:
                self.replay_storage.add(time_step)
            else:
                self.replay_cache.add(time_step)

            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='../diffusion_reward/configs/rl', config_name='default')
def main(cfgs):
    workspace = Workspace(cfgs)
    workspace.train()


if __name__ == '__main__':
    main()