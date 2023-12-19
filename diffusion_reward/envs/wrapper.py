import random
from collections import deque
from typing import Any, NamedTuple

import dm_env
import gym
import numpy as np
from dm_env import StepType, specs


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, img_size, frame_stack, action_repeat, mt1=None, render_size=256):
        super().__init__(env)
        self.env = env
        self._num_frames = frame_stack
        self._action_repeat = action_repeat
        self._frames = deque([], maxlen=self._num_frames)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._num_frames * 3, img_size, img_size),
            dtype=np.uint8,
        )
        self.action_space = self.env.action_space
        self._res = (img_size, img_size)
        self.img_size = img_size
        self.render_size = render_size
        self.mt1 = mt1

    def state(self):
        state = self._state_obs.astype(np.float32)
        return state

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)
    
    def _get_pixel_obs(self, pixel_obs):
        return pixel_obs[:, :, ::-1].transpose(
            2, 0, 1
        )
    
    def reset(self):
        self.env.set_task(self.mt1.train_tasks[random.randint(0, 49)])
        self._state_obs, info = self.env.reset()
        obs = self.env.render().transpose(2, 0, 1)
        return obs.copy(), info  

    def step(self, action):
        rewards = 0
        for _ in range(self._action_repeat):
            next_obs, _, trunc, termn, info = self.env.step(action)
            rewards += int(info['success']) 
        self._state_obs = next_obs
        next_obs = self.env.render().transpose(2, 0, 1).copy()
        return next_obs, rewards, False, info

    def render(self):
        return np.flipud(self.env.render().copy()).copy()

    def observation_spec(self):
        print(self.observation_space)
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def __getattr__(self, name):
        return getattr(self.env, name)



class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimitWrapper, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
    
    def render(self):
        return self.env.render()


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        obs, info = self._env.reset()
        return self._augment_time_step(obs, state=self.prop_state()) 

    def step(self, action):
        next_obs, reward, done, info = self._env.step(action)
        discount = 1.0
        is_success = info['success']
        return self._augment_time_step(next_obs,
                                       next_obs, 
                                       action,
                                       reward,
                                       is_success,
                                       discount,
                                       done)
    def prop_state(self):
        state = self._env.state()
        return np.concatenate((state[:4], state[18 : 18 + 4]))
    
    def _augment_time_step(self, obs, state, action=None, reward=None, success=False, discount=1.0, done=False):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
            reward = 0.0
            success = 0.0
            discount = 1.0
            done = False
        return ExtendedTimeStep(observation=obs, 
                                state=state, 
                                action=action,
                                reward=reward,
                                is_success=success, 
                                discount=discount,
                                done=done)
    
    def state_spec(self):
        return specs.BoundedArray((8,), np.float32, name='state', minimum=0, maximum=255)
    
    def observation_spec(self):
        return specs.BoundedArray(self._env.observation_space.shape, np.uint8, name='observation', minimum=0, maximum=255)

    def action_spec(self):
        return specs.BoundedArray(self._env.action_space.shape, np.float32, name='action', minimum=-1, maximum=1.0)

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    done: Any
    reward: Any
    discount: Any
    observation: Any
    state: Any
    action: Any
    is_success: Any

    def last(self):
        return self.done

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStepAdroit(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    observation_sensor: Any
    action: Any
    n_goal_achieved: Any
    time_limit_reached: Any
    is_success: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)