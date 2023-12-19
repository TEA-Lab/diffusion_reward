from pathlib import Path

from .amp import AMP
from .diffusion_reward import DiffusionReward
from .rnd import RND
from .viper import VIPER


def make_rm(cfg):
    if cfg.rm_model == 'diffusion_reward':
        cfg.cfg_path = str(Path(__file__).parents[3]) + cfg.cfg_path
        cfg.ckpt_path = str(Path(__file__).parents[3]) + cfg.ckpt_path
        rm = DiffusionReward(cfg=cfg)
    elif cfg.rm_model == 'viper':
        cfg.cfg_path = str(Path(__file__).parents[3]) + cfg.cfg_path
        cfg.ckpt_path = str(Path(__file__).parents[3]) + cfg.ckpt_path
        rm = VIPER(cfg=cfg)
    elif cfg.rm_model == 'amp':
        rm = AMP(cfg)
    elif cfg.rm_model == 'rnd':
        rm = RND(cfg)
    return rm