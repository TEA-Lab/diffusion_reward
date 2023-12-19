import diffusion_reward.envs.adroit as adroit
import diffusion_reward.envs.metaworld as metaworld
import metaworld.envs.mujoco.env_dict as _mw_envs

from .adroit import _mj_envs


def make_env(name, frame_stack, action_repeat, seed):
    if name in _mj_envs:
        env = adroit.make(name, frame_stack, action_repeat, seed)
    elif name in _mw_envs.ALL_V2_ENVIRONMENTS.keys():
        env = metaworld.make(name, frame_stack, action_repeat, seed)
    else:
        raise NotImplementedError
    return env