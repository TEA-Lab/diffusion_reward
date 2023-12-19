import numpy as np
import torch
import torch.nn as nn
from torch import autograd


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        with torch.no_grad():
            x = torch.ones((3, 64,64))
            x = self.convnet(x)
        self.repr_dim = np.prod(x.shape)

        self.trunk = nn.Sequential(
            nn.Linear(self.repr_dim, args.hidden_dim), nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim), nn.Tanh(),
            nn.Linear(args.hidden_dim, 1))

        self.apply(weight_init)

    def compute_grad_pen(self,
                         expert_state,
                         policy_state,
                         lambda_=0.1):
        alpha = torch.rand(expert_state.size(0), 1, 1, 1)
        alpha = alpha.expand_as(expert_state).to(expert_state.device)

        mixup_data = alpha * expert_state + (1 - alpha) * policy_state
        mixup_data.requires_grad = True

        disc = self(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def forward(self, obs):
        x = self.convnet(obs)
        x = x.reshape(x.shape[0], -1)
        x = self.trunk(x)
        return x


class AMP(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.disc = Discriminator(args)
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=args.disc_lr)

        # set attribute
        for attr_name, attr_value in args.items():
            print(attr_name, attr_value)
            setattr(self, attr_name, attr_value)

    def update(self, batch, expert_obs):
        metrics = dict()

        obs, _, _, _, _ = to_torch(batch, self.device) 
        obs = obs[:self.batch_size] / 127.5 - 1.0
        expert_obs = torch.as_tensor(expert_obs).to(self.device).permute(0, 3, 1, 2)
        expert_obs = expert_obs[:self.batch_size] / 127.5 - 1.0

        policy_d = self.disc(obs)
        expert_d = self.disc(expert_obs)

        expert_loss = (expert_d - 1).pow(2).mean()
        policy_loss = (policy_d + 1).pow(2).mean()

        gail_loss = expert_loss + policy_loss
        grad_pen = self.disc.compute_grad_pen(expert_obs, obs)

        loss = gail_loss + grad_pen

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()

        metrics['disc_expert_loss'] = expert_loss.item()
        metrics['disc_policy_loss'] = policy_loss.item()
        metrics['grad_pen'] = grad_pen.item()
        return metrics

    @torch.no_grad()
    def calc_expl_reward(self, obs, next_obs):
        obs = (obs / 127.5 - 1.0).float()
        feat = self.disc(obs)

        rewards = torch.clamp(1 - 0.25 * torch.square(feat - 1), min=0) * self.expl_scale
        return rewards