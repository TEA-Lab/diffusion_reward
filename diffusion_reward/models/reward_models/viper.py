from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

from ..video_models.videogpt.transformer import VideoGPTTransformer


class VIPER(nn.Module):
    def __init__(self, cfg):
        super(VIPER, self).__init__()

        # load video models
        self.model_cfg = OmegaConf.load(cfg.cfg_path)
        self.model = VideoGPTTransformer(self.model_cfg)
        self.model.load_state_dict(torch.load(cfg.ckpt_path))
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad = False

        # set attribute
        for attr_name, attr_value in cfg.items():
            setattr(self, attr_name, attr_value)
        
        # standardization
        self.use_std = cfg.use_std
        if self.use_std:
            stat_path = str(Path(__file__).parents[3]) + cfg.stat_path
            with open(stat_path, 'r') as file:
                self.stat = yaml.safe_load(file)[cfg.task_name]

        # build exploration reward model
        self.use_expl_reward = cfg.use_expl_reward
        if self.use_expl_reward:
            cfg.expl_reward.obs_shape = cfg.obs_shape
            cfg.expl_reward.action_shape = cfg.action_shape
            self.expl_reward = hydra.utils.instantiate(cfg.expl_reward)
            self.expl_scale = cfg.expl_scale

    def imgs_to_batch(self, x, reward_type='likelihood'):
        '''
        input:
            imgs: B * T * H * W * C
            (mostly): 1 * T * ...
        '''
        seq_len = x.shape[1]
        num_frames = self.model_cfg.num_frames + 1
        n_skip = self.model_cfg.frame_skip
        subseq_len = num_frames * n_skip

        x = x.permute(0, 1, 4, 2 ,3)
        embs, indices = self.model.encode_to_z(x)
        indices = indices.reshape(indices.shape[0], seq_len, -1)
        embs = embs.reshape(embs.shape[0], seq_len, indices.shape[-1], -1)
        
        if reward_type == 'likelihood':
            post_idxes = list(range(seq_len - subseq_len + 1))
            batch_indices = [indices[:, idx:idx+subseq_len:n_skip] for idx in post_idxes]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1)
            batch_embs = [embs[:, idx:idx+subseq_len:n_skip] for idx in post_idxes]
            batch_embs = torch.stack(batch_embs, dim=0)
            batch_embs = batch_embs.squeeze(1).reshape(batch_embs.shape[0], -1, batch_embs.shape[-1])

            pre_batch_indices = [indices[:, idx].tile((1, num_frames)) for idx in range(subseq_len-1)]
            pre_batch_indices = torch.concat(pre_batch_indices, dim=0)
            batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0)

            pre_batch_embs = [embs[:, idx].tile((1, num_frames, 1)) for idx in range(subseq_len-1)]
            pre_batch_embs = torch.concat(pre_batch_embs, dim=0)
            batch_embs = torch.concat([pre_batch_embs, batch_embs], dim=0)
        elif reward_type == 'entropy':
            post_idxes = list(range(seq_len - subseq_len + 2))
            batch_indices = [indices[:, idx:idx+subseq_len-n_skip:n_skip] for idx in post_idxes]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1)
            batch_embs = [embs[:, idx:idx+subseq_len-n_skip:n_skip] for idx in post_idxes]
            batch_embs = torch.stack(batch_embs, dim=0)
            batch_embs = batch_embs.squeeze(1).reshape(batch_embs.shape[0], -1, batch_embs.shape[-1])

            pre_batch_indices = [indices[:, idx].tile((1, num_frames-1)) for idx in range(subseq_len-2)]
            pre_batch_indices = torch.concat(pre_batch_indices, dim=0)
            batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0)

            pre_batch_embs = [embs[:, idx].tile((1, num_frames-1, 1)) for idx in range(subseq_len-2)]
            pre_batch_embs = torch.concat(pre_batch_embs, dim=0)
            batch_embs = torch.concat([pre_batch_embs, batch_embs], dim=0)
        else:
            raise NotImplementedError

        return batch_embs, batch_indices

    @torch.no_grad()
    def calc_reward(self, imgs):
        batch_embs, batch_indices = self.imgs_to_batch(imgs, self.reward_type)
        sos_tokens = self.model.calc_sos_tokens(imgs, batch_embs).tile((batch_embs.shape[0], 1, 1))

        rewards = self.cal_log_prob(batch_embs, batch_indices, sos_tokens, target_indices=batch_indices, reward_type=self.reward_type)
        return rewards    

    @torch.no_grad()
    def cal_log_prob(self, embs, x, c, target_indices=None, reward_type='likelihood'):
        self.model.eval()
        if not self.model.use_vqemb:
            x = torch.cat((c, x), dim=1) if x is not None else c   
        else:
            x = torch.cat((c, embs), dim=1) if x is not None else c

        logits, _ = self.model.transformer(x[:, :-1])
        probs = F.log_softmax(logits, dim=-1)

        if reward_type == 'likelihood':
            target = F.one_hot(target_indices, num_classes=self.model_cfg.codec.num_codebook_vectors)
            if self.compute_joint:
                rewards = (probs * target).sum(-1).sum(-1, keepdim=True)
            else:
                num_valid_logits = int(logits.shape[1] // (self.model_cfg.num_frames + 1))
                rewards = (probs * target).sum(-1)[:, -num_valid_logits:].sum(-1, keepdim=True)
        elif reward_type == 'entropy':
            num_valid_logits = int(logits.shape[1] // (self.model_cfg.num_frames))
            entropy = (- probs * probs.exp()).sum(-1)[:, -num_valid_logits:].sum(-1, keepdim=True)
            rewards = - entropy
        else:
            raise NotImplementedError

        if self.use_std:
            rewards_std = (rewards - self.stat[0]) / self.stat[1]
        scaled_rewards = (1 - self.expl_scale) * rewards_std
        return scaled_rewards

    def update(self, batch):
        metrics = dict()

        if self.use_expl_reward:
            metrics.update(self.expl_reward.update(batch))
        return metrics

    @torch.no_grad()
    def calc_expl_reward(self, obs, next_obs):
        expl_rewards = self.expl_reward.calc_reward(obs, next_obs) * self.expl_scale    
        return expl_rewards 