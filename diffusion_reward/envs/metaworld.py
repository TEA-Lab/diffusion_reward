from metaworld import MT1

from .wrapper import (ExtendedTimeStepWrapper, MetaWorldWrapper,
                      TimeLimitWrapper)


def mw_gym_make(task_name, task_id=0, seed=None):
    if seed is not None:
        mt1 = MT1(task_name, seed=seed) 
    else:
        mt1 = MT1(task_name) # Construct the benchmark, sampling tasks
    env = mt1.train_classes[task_name](render_mode='rgb_array')

    if task_id is not None:
        env.set_task(mt1.train_tasks[task_id]) 
    return env, mt1


def make(name, frame_stack, action_repeat, seed, img_size=64, episode_length=100, task_id=0): # TODO change here or reset???
    env, mt1 = mw_gym_make(name, task_id=task_id, seed=seed)

    env = MetaWorldWrapper(env, img_size, frame_stack, action_repeat, mt1=mt1)
    env = TimeLimitWrapper(env, max_episode_steps=episode_length)
    env = ExtendedTimeStepWrapper(env)
    #env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    return env

