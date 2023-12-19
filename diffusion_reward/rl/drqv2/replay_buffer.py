import datetime
import io
import os
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayCache:
    def __init__(self, device=None):
        self.images = []
        self.time_steps = []
        self.device = device
        
    def add(self, time_step):
        self.images.append(time_step.observation)
        self.time_steps.append(time_step)

    def pop(self, replay_storage, rm):
        self.images = np.array(self.images)
        self.images = torch.Tensor(self.images).to(self.device).permute(0, 2, 3, 1)[None] / 127.5 - 1.0

        rewards = rm.calc_reward(self.images).cpu().numpy()

        # reward replay: None for eval only
        if replay_storage is not None:
            for i, ts in enumerate(self.time_steps):
                env_reward = ts.reward
                if rm.use_env_reward:
                    ts = ts._replace(reward=rewards[i] + env_reward)
                else:
                    ts = ts._replace(reward=rewards[i])

                replay_storage.add(ts)

        self.images = []
        self.time_steps = []
        return np.sum(rewards)


class AMPBuffer:
    def __init__(self, max_size=1e5, batch_size=64, cfg=None) -> None:
        self._max_size = int(max_size)
        self.obses = None
        self.count = 0
        self.full = False
        self.batch_size = batch_size
        self.cfg = cfg
        self.init_demo()

    def add(self, images):
        if self.obses is None:
            self.obses = np.empty((self._max_size, *images.shape), dtype=np.uint8)
        idx = self.count % self._max_size
        self.obses[idx] = images
        self.count += 1

        if self.count >= self._max_size:
            self.full = True

    def sample(self):
        idx = np.random.randint(0, self._max_size if self.full else self.count, size=self.batch_size, dtype=np.int)
        obses = self.obses[idx]
        return obses

    def init_demo(self):
        '''Load video dataset'''
        from PIL import Image

        root_dir = f'/home/taohuang/project/qizhi/diffusion-as-reward/video_dataset/{self.cfg.domain}'
        task = self.cfg.task_name
        task_path = os.path.join(root_dir, task, 'train') 
        task_videos = [os.path.join(task_path, f'{i}') for i in range(len(os.listdir(task_path)))]
        for video_path in task_videos:
            frames = os.listdir(video_path)
            frame_path = [os.path.join(video_path, f'{i}.png') for i in range(len(frames))]
            for frame in frame_path:
                self.add(np.array(Image.open(frame)).astype(np.uint8))


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()

    def update_nstep(self, new_nstep):
        self._nstep = new_nstep

    def update_discount(self, new_discount):
        self._discount = new_discount


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader, iterable
