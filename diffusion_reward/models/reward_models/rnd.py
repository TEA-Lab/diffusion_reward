import hydra
import torch
import torch.nn as nn


class RND(nn.Module):
    def __init__(self, cfg):
        super(RND, self).__init__()

        # set attribute
        for attr_name, attr_value in cfg.items():
            setattr(self, attr_name, attr_value)

        # build exploration reward model
        self.use_expl_reward = cfg.use_expl_reward
        assert self.use_expl_reward is True

        cfg.expl_reward.obs_shape = cfg.obs_shape
        cfg.expl_reward.action_shape = cfg.action_shape
        self.expl_reward = hydra.utils.instantiate(cfg.expl_reward)
        self.expl_scale = cfg.expl_scale

    @torch.no_grad()
    def calc_reward(self, imgs):
        zero_rewards = torch.zeros((imgs.shape[1])).unsqueeze(1)
        return zero_rewards

    def update(self, batch):
        metrics = dict()
        metrics.update(self.expl_reward.update(batch))
        return metrics

    @torch.no_grad()
    def calc_expl_reward(self, obs, next_obs):
        expl_rewards = self.expl_reward.calc_reward(obs, next_obs) * self.expl_scale    
        return expl_rewards 