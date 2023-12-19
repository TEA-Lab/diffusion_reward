# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

from collections import deque

import gym
import mj_envs
import numpy as np
import torch
from diffusion_reward.envs.wrapper import ExtendedTimeStepAdroit
from dm_env import StepType, specs
from mjrl.utils.gym_env import GymEnv
from PIL import Image

_mj_envs = {'pen-v0', 'hammer-v0', 'door-v0', 'relocate-v0'}


class BasicAdroitEnv(gym.Env): # , ABC
    def __init__(self, env, cameras, latent_dim=512, hybrid_state=True, channels_first=False, 
    height=64, width=64, test_image=False, num_repeats=1, num_frames=1, encoder_type=None, device=None):
        self._env = env
        self.env_id = env.env.unwrapped.spec.id
        self.device = device

        self._num_repeats = num_repeats
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)

        self.encoder = None
        self.transforms = None
        self.encoder_type = encoder_type

        if test_image:
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
        self.test_image = test_image

        self.cameras = cameras
        self.latent_dim = latent_dim
        self.hybrid_state = hybrid_state
        self.channels_first = channels_first
        self.height = height
        self.width = width
        self.action_space = self._env.action_space
        self.env_kwargs = {'cameras' : cameras, 'latent_dim' : latent_dim, 'hybrid_state': hybrid_state,
                           'channels_first' : channels_first, 'height' : height, 'width' : width}

        shape = [3, self.width, self.height]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self.sim = env.env.sim
        self._env.spec.observation_dim = latent_dim

        if hybrid_state :
            if self.env_id in _mj_envs:
                self._env.spec.observation_dim += 24 # Assuming 24 states for adroit hand.

        self.spec = self._env.spec
        self.observation_dim = self.spec.observation_dim
        self.horizon = self._env.env.spec.max_episode_steps

    def get_obs(self,):
        # for our case, let's output the image, and then also the sensor features
        if self.env_id in _mj_envs :
            env_state = self._env.env.get_env_state()
            qp = env_state['qpos']

        if self.env_id == 'pen-v0':
            qp = qp[:-6]
        elif self.env_id == 'door-v0':
            qp = qp[4:-2]
        elif self.env_id == 'hammer-v0':
            qp = qp[2:-7]
        elif self.env_id == 'relocate-v0':
            qp = qp[6:-6]

        imgs = [] # number of image is number of camera

        if self.encoder is not None:
            for cam in self.cameras :
                img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0)
                # img = env.env.sim.render(width=64, height=64, mode='offscreen')
                img = img[::-1, :, : ] # Image given has to be flipped
                if self.channels_first :
                    img = img.transpose((2, 0, 1))
                #img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img = self.transforms(img)
                imgs.append(img)

            inp_img = torch.stack(imgs).to(self.device) # [num_cam, C, H, W]
            z = self.encoder.get_features(inp_img).reshape(-1)
            # assert z.shape[0] == self.latent_dim, "Encoded feature length : {}, Expected : {}".format(z.shape[0], self.latent_dim)
            pixels = z
        else: # true
            if not self.test_image:
                for cam in self.cameras : # for each camera, render once
                    img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
                    # img = img[::-1, :, : ] # Image given has to be flipped
                    if self.channels_first :
                        img = img.transpose((2, 0, 1)) # then it's 3 x width x height
                    # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
                    #img = img.astype(np.uint8)
                    # img = Image.fromarray(img) # TODO is this necessary?
                    imgs.append(img)
            else: # true
                img = (np.random.rand(1, self.image_size[0], self.image_size[1]) * 255).astype(np.uint8)
                imgs.append(img)
            pixels = np.concatenate(imgs, axis=0)

        # TODO below are what we originally had... 
        # if not self.test_image:
        #     for cam in self.cameras : # for each camera, render once
        #         img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
        #         # img = img[::-1, :, : ] # Image given has to be flipped
        #         if self.channels_first :
        #             img = img.transpose((2, 0, 1)) # then it's 3 x width x height
        #         # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
        #         #img = img.astype(np.uint8)
        #         # img = Image.fromarray(img) # TODO is this necessary?
        #         imgs.append(img)
        # else:
        #     img = (np.random.rand(1, 64, 64) * 255).astype(np.uint8)
        #     imgs.append(img)
        # pixels = np.concatenate(imgs, axis=0)

        if not self.hybrid_state : # this defaults to True... so RRL uses hybrid state
            qp = None

        sensor_info = qp
        return pixels, sensor_info

    def get_env_infos(self):
        return self._env.get_env_infos()

    def set_seed(self, seed):
        return self._env.set_seed(seed)

    def get_stacked_pixels(self): #TODO fix it
        assert len(self._frames) == self._num_frames
        stacked_pixels = np.concatenate(list(self._frames), axis=0)
        return stacked_pixels

    def reset(self):
        self._env.reset()
        pixels, sensor_info = self.get_obs()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return stacked_pixels, sensor_info

    def get_obs_for_first_state_but_without_reset(self):
        pixels, sensor_info = self.get_obs()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return stacked_pixels, sensor_info

    def step(self, action):
        reward_sum = 0.0
        discount_prod = 1.0 # TODO pen can terminate early 
        n_goal_achieved = 0
        for i_action in range(self._num_repeats): 
            obs, reward, done, env_info = self._env.step(action)
            reward = bool(env_info['goal_achieved']) #- 1
            reward_sum += reward 
            if env_info['goal_achieved'] == True:
                n_goal_achieved += 1
            if done:
                break
        env_info['n_goal_achieved'] = n_goal_achieved
        # now get stacked frames
        pixels, sensor_info = self.get_obs()
        self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return [stacked_pixels, sensor_info], reward_sum, done, env_info

    def set_env_state(self, state):
        return self._env.set_env_state(state)
        
    def get_env_state(self):
        return self._env.get_env_state(state)

    def evaluate_policy(self, policy,
    					num_episodes=5,
    					horizon=None,
    					gamma=1,
    					visual=False,
    					percentile=[],
    					get_full_dist=False,
    					mean_action=False,
    					init_env_state=None,
    					terminate_at_done=True,
    					seed=123):
        # TODO this needs to be rewritten

        self.set_seed(seed)
        horizon = self.horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)
        self.encoder.eval()

        for ep in range(num_episodes):
            o = self.reset()
            if init_env_state is not None:
                self.set_env_state(init_env_state)
            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                self.render() if visual is True else None
                o = self.get_obs(self._env.get_obs())
                a = policy.get_action(o)[1]['evaluation'] if mean_action is True else policy.get_action(o)[0]
                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        full_dist = ep_returns if get_full_dist is True else None

        return [base_stats, percentile_stats, full_dist]

    def get_pixels_with_width_height(self, w, h):
        imgs = [] # number of image is number of camera

        for cam in self.cameras : # for each camera, render once
            img = self._env.env.sim.render(width=w, height=h, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
            # img = img[::-1, :, : ] # Image given has to be flipped
            if self.channels_first :
                img = img.transpose((2, 0, 1)) # then it's 3 x width x height
            # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
            #img = img.astype(np.uint8)
            # img = Image.fromarray(img) # TODO is this necessary?
            imgs.append(img)

        pixels = np.concatenate(imgs, axis=0)
        return pixels


class AdroitEnv:
    # a wrapper class that will make Adroit env looks like a dmc env
    def __init__(self, env_name, test_image=False, cam_list=None, image_size=(64, 64),
        num_repeats=2, num_frames=1, env_feature_type='pixels', device=None, reward_rescale=False): 
        default_env_to_cam_list = {
            'hammer-v0': ['top'],
            'door-v0': ['top'],
            'pen-v0': ['vil_camera'],
            'relocate-v0': ['cam1', 'cam2', 'cam3',],
        }
        if cam_list is None:
            cam_list = default_env_to_cam_list[env_name]
        self.env_name = env_name
        reward_rescale_dict = {
            'hammer-v0': 1/100,
            'door-v0': 1/20,
            'pen-v0': 1/50,
            'relocate-v0': 1/30,
        }
        if reward_rescale:
            self.reward_rescale_factor = reward_rescale_dict[env_name]
        else:
            self.reward_rescale_factor = 1

        # env, _ = make_basic_env(env_name, cam_list=cam_list, from_pixels=from_pixels, hybrid_state=True, 
        #     test_image=test_image, channels_first=True, num_repeats=num_repeats, num_frames=num_frames)
        env = GymEnv(env_name)
        if env_feature_type == 'state':
            raise NotImplementedError("state env not ready")
        elif env_feature_type == 'resnet18' or env_feature_type == 'resnet34' :
            # TODO maybe we will just throw everything into it.. 
            height = 256
            width = 256
            latent_dim = 512
            env = BasicAdroitEnv(env, cameras=cam_list,
                height=height, width=width, latent_dim=latent_dim, hybrid_state=True, 
                test_image=test_image, channels_first=False, num_repeats=num_repeats, num_frames=num_frames, encoder_type=env_feature_type, 
                device=device
                )
        elif env_feature_type == 'pixels':
            height = image_size[0]
            width = image_size[1]
            latent_dim = height*width*len(cam_list)*num_frames
            # RRL class instance is environment wrapper...
            env = BasicAdroitEnv(env, cameras=cam_list,
                height=height, width=width, latent_dim=latent_dim, hybrid_state=True, 
                test_image=test_image, channels_first=True, num_repeats=num_repeats, num_frames=num_frames, device=device)
        else:
            raise ValueError("env feature not supported")

        self._env = env
        self.obs_dim = env.spec.observation_dim
        self.obs_sensor_dim = 24
        self.act_dim = env.spec.action_dim
        self.horizon = env.spec.horizon
        self.image_size = image_size
        number_channel = len(cam_list) * 3 * num_frames

        if env_feature_type == 'pixels':
            self._obs_spec = specs.BoundedArray(shape=(number_channel, image_size[0], image_size[1]), dtype='uint8', name='observation', minimum=0, maximum=255)
            self._obs_sensor_spec = specs.Array(shape=(self.obs_sensor_dim,), dtype='float32', name='observation_sensor')
        elif env_feature_type == 'resnet18' or env_feature_type == 'resnet34' :
            self._obs_spec = specs.Array(shape=(512 * num_frames *len(cam_list) ,), dtype='float32', name='observation') # TODO fix magic number 
            self._obs_sensor_spec = specs.Array(shape=(self.obs_sensor_dim,), dtype='float32', name='observation_sensor')
        self._action_spec = specs.BoundedArray(shape=(self.act_dim,), dtype='float32', name='action', minimum=-1.0, maximum=1.0)

    def reset(self):
        # pixels and sensor values
        obs_pixels, obs_sensor = self._env.reset()
        obs_sensor = obs_sensor.astype(np.float32)
        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        time_step = ExtendedTimeStepAdroit(observation=obs_pixels,
                                     observation_sensor=obs_sensor,
                                step_type=StepType.FIRST,
                                action=action,
                                reward=0.0,
                                discount=1.0,
                                n_goal_achieved=0,
                                time_limit_reached=False,
                                is_success=False)
        return time_step

    def get_current_obs_without_reset(self):
        # use this to obtain the first state in a demo
        obs_pixels, obs_sensor = self._env.get_obs_for_first_state_but_without_reset()
        obs_sensor = obs_sensor.astype(np.float32)
        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        time_step = ExtendedTimeStepAdroit(observation=obs_pixels,
                                     observation_sensor=obs_sensor,
                                step_type=StepType.FIRST,
                                action=action,
                                reward=0.0,
                                discount=1.0,
                                n_goal_achieved=0,
                                time_limit_reached=False,
                                is_success=False)
        return time_step

    def get_pixels_with_width_height(self, w, h):
        return self._env.get_pixels_with_width_height(w, h)

    def step(self, action, force_step_type=None, debug=False):
        obs_all, reward, done, env_info = self._env.step(action)
        obs_pixels, obs_sensor = obs_all
        obs_sensor = obs_sensor.astype(np.float32)

        discount = 1.0
        n_goal_achieved = env_info['n_goal_achieved']
        time_limit_reached = env_info['TimeLimit.truncated'] if 'TimeLimit.truncated' in env_info else False
        if done:
            steptype = StepType.LAST
        else:
            steptype = StepType.MID

        if done and not time_limit_reached:
            discount = 0.0

        if force_step_type is not None:
            if force_step_type == 'mid':
                steptype = StepType.MID
            elif force_step_type == 'last':
                steptype = StepType.LAST
            else:
                steptype = StepType.FIRST

        reward = reward * self.reward_rescale_factor

        time_step = ExtendedTimeStepAdroit(observation=obs_pixels,
                                     observation_sensor=obs_sensor,
                                step_type=steptype,
                                action=action,
                                reward=reward,
                                discount=discount,
                                n_goal_achieved=n_goal_achieved,
                                time_limit_reached=time_limit_reached,
                                is_success=bool(n_goal_achieved))

        if debug:
            return obs_all, reward, done, env_info
        return time_step

    def observation_spec(self):
        return self._obs_spec

    def observation_sensor_spec(self):
        return self._obs_sensor_spec

    def action_spec(self):
        return self._action_spec

    def set_env_state(self, state):
        self._env.set_env_state(state)

    # def __getattr__(self, name):
    #     return getattr(self, name)
    def render(self):
        return self.get_pixels_with_width_height(256, 256).transpose(1, 2, 0)


def make(name, frame_stack, action_repeat, seed):
    env = AdroitEnv(
        env_name=name,
        num_frames=frame_stack,
        num_repeats=action_repeat,
    )
    return env
