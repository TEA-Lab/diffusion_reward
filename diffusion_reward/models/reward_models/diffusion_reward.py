import os
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..video_models.vqdiffusion.modeling.build import build_model
from ..video_models.vqdiffusion.modeling.transformers.diffusion_transformer import (
    index_to_log_onehot, log_categorical, log_onehot_to_index,
    sum_except_batch)
from ..video_models.vqdiffusion.utils.io import load_yaml_config
from ..video_models.vqdiffusion.utils.misc import get_model_parameters_info


class DiffusionReward(nn.Module):
    def __init__(self, cfg):
        super(DiffusionReward, self).__init__()

        # load video models
        self.info = self.get_model(ema=True, model_path=cfg.ckpt_path, config_path=cfg.cfg_path)
        self.model = self.info['model']
        self.epoch = self.info['epoch']
        self.model_name = self.info['model_name']
        # self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad = False

        # set attribute
        for attr_name, attr_value in cfg.items():
            print(attr_name, attr_value)
            setattr(self, attr_name, attr_value)
        
        # standardization
        self.use_std = cfg.use_std
        if self.use_std:
            stat_path = str(Path(__file__).parents[3]) + cfg.stat_path
            with open(stat_path, 'r') as file:
                self.stat = yaml.safe_load(file)[cfg.task_name][cfg.skip_step]

        # build exploration reward model
        self.use_expl_reward = cfg.use_expl_reward
        if self.use_expl_reward:
            cfg.expl_reward.obs_shape = cfg.obs_shape
            cfg.expl_reward.action_shape = cfg.action_shape
            self.expl_reward = hydra.utils.instantiate(cfg.expl_reward)
            self.expl_scale = cfg.expl_scale

    def get_model(self, ema, model_path, config_path):
        if 'OUTPUT' in model_path: # pretrained model
            model_name = model_path.split(os.path.sep)[-3]
        else: 
            model_name = os.path.basename(config_path).replace('.yaml', '')

        config = load_yaml_config(config_path)

        model = build_model(config)
        model_parameters = get_model_parameters_info(model)
        
        print(model_parameters)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
            if 'last_epoch' in ckpt:
                epoch = ckpt['last_epoch']
            elif 'epoch' in ckpt:
                epoch = ckpt['epoch']
            else:
                epoch = 0

            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            print('Model missing keys:\n', missing)
            print('Model unexpected keys:\n', unexpected)

            if ema==True and 'ema' in ckpt:
                print("Evaluate EMA model")
                ema_model = model.get_ema_model()
                missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)
        else:
            epoch = None
        return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}

    def imgs_to_batch(self, x, reward_type='entropy'):
        '''
        input:
            imgs: B * T * H * W * C
            (mostly): 1 * T * ...
        '''
        assert x.max() <= 1
        # preprocessing
        seq_len = x.shape[1]
        num_frames = self.model.cfg.params['condition_emb_config']['params']['num_cond_frames']
        n_skip = self.model.frame_skip
        subseq_len = (num_frames + 1) * n_skip

        x = x.permute(0, 1, 4, 2 ,3)
        _, indices = self.model.content_codec.encode_to_z(x)
        assert indices.shape[0] == 1
        indices = indices.reshape(indices.shape[0], seq_len, -1)

        if reward_type == 'entropy':
            # only return conditional frames
            post_idxes = list(range(seq_len - subseq_len + 2))
            batch_indices = [indices[:, idx:idx+subseq_len-n_skip:n_skip] for idx in post_idxes]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1)    
            
            if subseq_len - 2 > 0:
                pre_batch_indices = [indices[:, idx].tile((1, num_frames)) for idx in range(subseq_len-2)]
                pre_batch_indices = torch.concat(pre_batch_indices, dim=0)
                batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0)
            cond = {'condition_token': batch_indices}
        elif reward_type == 'likelihood':
            # return conditional frames + current frame
            post_idxes = list(range(seq_len - subseq_len + 1))
            batch_indices = [indices[:, idx:idx+subseq_len-n_skip:n_skip] for idx in post_idxes]
            batch_indices = torch.stack(batch_indices, dim=0)
            batch_indices = batch_indices.squeeze(1).reshape(batch_indices.shape[0], -1)    
            
            if subseq_len - 2 > 0:
                pre_batch_indices = [indices[:, idx].tile((1, num_frames)) for idx in range(subseq_len-1)]
                pre_batch_indices = torch.concat(pre_batch_indices, dim=0)
                batch_indices = torch.concat([pre_batch_indices, batch_indices], dim=0)
            cond = {'condition_token': batch_indices}
        else:
            raise NotImplementedError

        x = x.flatten(0, 1)
        cont = {'content_token': indices[0]}
        return cont, cond, indices[0]

    @torch.no_grad()
    def calc_reward(self, imgs):
        self.model.eval()
        content, condition, _ = self.imgs_to_batch(imgs, reward_type=self.reward_type)
        content_token = content['content_token']
        condition_token = condition['condition_token']

        rewards = self.calc_vlb(content_token, condition_token)
        if self.use_std:
            rewards_std = (rewards - self.stat[0]) / self.stat[1]
        scaled_rewards = (1 - self.expl_scale) * rewards_std
        return scaled_rewards    

    @torch.no_grad()
    def calc_vlb(self, cont_emb, cond_emb):
        x = cont_emb
        b, device = x.size(0), x.device
        transformer = self.model.transformer
        cond_emb = transformer.condition_emb(cond_emb).float()

        # t=0
        start_step = transformer.num_timesteps
        x_start = x
        t = torch.full((b,), start_step-1, device=device, dtype=torch.long)
        log_x_start = index_to_log_onehot(x_start, transformer.num_classes)

        # t=T
        zero_logits = torch.zeros((b, transformer.num_classes-1, transformer.shape),device=device)
        one_logits = torch.ones((b, 1, transformer.shape),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)

        # denoised time_steps
        diffusion_list = [index for index in range(start_step-1, -1, -1-self.skip_step)]
        if diffusion_list[-1] != 0:
            diffusion_list.append(0)

        vlbs = []
        if self.reward_type == 'entropy':
            # use denoised samples for estimation
            for _ in range(self.num_sample):
                start_step = transformer.num_timesteps
                x_start = x
                t = torch.full((b,), start_step-1, device=device, dtype=torch.long)
                log_x_start = index_to_log_onehot(x_start, transformer.num_classes)

                # t=T
                zero_logits = torch.zeros((b, transformer.num_classes-1, transformer.shape),device=device)
                one_logits = torch.ones((b, 1, transformer.shape),device=device)
                mask_logits = torch.cat((zero_logits, one_logits), dim=1)
                log_z = torch.log(mask_logits)

                model_log_probs = []
                log_zs = []
                ts = []
                vlb = []
                for diffusion_index in diffusion_list:
                    t = torch.full((b,), diffusion_index, device=device, dtype=torch.long)
                    log_x_recon = transformer.cf_predict_start(log_z, cond_emb, t)
                    log_zs.append(log_z)
                    if diffusion_index > self.skip_step:
                        model_log_prob = transformer.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t-self.skip_step)
                        ts.append(t-self.skip_step)
                    else:
                        model_log_prob = transformer.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t)
                        ts.append(t)

                    model_log_probs.append(model_log_prob)
                    log_z = transformer.log_sample_categorical(model_log_prob, noise=self.noise, noise_scale=self.noise_scale)

                x_start = log_onehot_to_index(log_z)
                log_x_start = index_to_log_onehot(x_start, transformer.num_classes)
                for i, model_log_prob in enumerate(model_log_probs[:-1]):
                    log_true_prob = transformer.q_posterior(log_x_start=log_x_start, log_x_t=log_zs[i], t=ts[i])
                    kl = transformer.multinomial_kl(log_true_prob, model_log_prob)
                    kl = sum_except_batch(kl).unsqueeze(1)
                    vlb.append(-kl)

                log_probs = model_log_probs[-1].permute(0, 2, 1)
                target = F.one_hot(x_start, num_classes=transformer.num_classes)
                rewards = (log_probs * target).sum(-1).sum(-1)
                rewards += torch.concat(vlb, dim=1).sum(dim=1)
                vlbs.append(rewards)
        elif self.reward_type == 'likelihood':
            # use observed samples for estimation
            for diffusion_index in diffusion_list:
                t = torch.full((b,), diffusion_index, device=device, dtype=torch.long)
                log_x_recon = transformer.cf_predict_start(log_z, cond_emb, t)
                if diffusion_index > self.skip_step:
                    model_log_prob = transformer.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t-self.skip_step)
                    log_true_prob = transformer.q_posterior(log_x_start=log_x_start, log_x_t=log_z, t=t-self.skip_step)
                else:
                    model_log_prob = transformer.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t)
                    log_true_prob = transformer.q_posterior(log_x_start=log_x_start, log_x_t=log_z, t=t)

                log_z = transformer.log_sample_categorical(model_log_prob, noise=self.noise, noise_scale=self.noise_scale)

                # -KL if t !=0 else LL
                if diffusion_index != 0:
                    kl = transformer.multinomial_kl(log_true_prob, model_log_prob)
                    kl = sum_except_batch(kl).unsqueeze(1)
                    vlbs.append(-kl)
                else:
                    decoder_ll = log_categorical(log_x_start, model_log_prob)
                    decoder_ll = sum_except_batch(decoder_ll).unsqueeze(1)   
                    vlbs.append(decoder_ll)

        else:
            raise NotImplementedError

        rewards = torch.stack(vlbs, dim=1).mean(1)
        return rewards

    def update(self, batch):
        metrics = dict()

        if self.use_expl_reward:
            metrics.update(self.expl_reward.update(batch))
        return metrics

    @torch.no_grad()
    def calc_expl_reward(self, obs, next_obs):
        expl_rewards = self.expl_reward.calc_reward(obs, next_obs) * self.expl_scale    
        return expl_rewards 