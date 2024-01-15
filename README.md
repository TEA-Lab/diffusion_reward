# Diffusion Reward
[[Project Website]](https://diffusion-reward.github.io/)
[[Paper]](https://arxiv.org/abs/2312.14134)
[[Data]](https://huggingface.co/datasets/tauhuang/diffusion_reward/tree/main)
[[Models]](https://huggingface.co/tauhuang/diffusion_reward/tree/main)

This is the official PyTorch implementation of the paper "[**Diffusion Reward: Learning Rewards via Conditional Video Diffusion**]()" by 

[Tao Huang](https://taohuang13.github.io/)<sup>\*</sup>, [Guangqi Jiang](https://lucca-cherries.github.io/)<sup>\*</sup>, [Yanjie Ze](https://yanjieze.com/), [Huazhe Xu](http://hxu.rocks/).

<p align="left">
  <img width="99%" src="docs/diffusion_reward_overview.png">
</p>

# 🛠️ Installation Instructions
Clone this repository.
```bash
git clone https://github.com/TaoHuang13/diffusion_reward.git
cd diffusion_reward
```
Create a virtual environment.
```bash
conda env create -f conda_env.yml 
conda activate diffusion_reward
pip install -e .
```
Install extra dependencies.
- Install PyTorch.
```bash
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```
- Install mujoco210 and mujoco-py following instructions [here](https://github.com/openai/mujoco-py#install-mujoco).

- Install Adroit dependencies.
```bash
cd env_dependencies
pip install -e mj_envs/.
pip install -e mjrl/.
cd ..
```
- Install MetaWorld following instructions [here](https://github.com/Farama-Foundation/Metaworld?tab=readme-ov-file#installation).

# 💻 Reproducing Experimental Results
## Download Video Demonstrations 
|Domain        | Tasks | Episodes| Size | Collection | Link  |       
|:------------- |:-------------:|:-----:|:----:|:-----:|:-----:|
| Adroit | 3 | 150 | 23.8M | [VRL3](https://github.com/microsoft/VRL3) | [Download](https://huggingface.co/datasets/tauhuang/diffusion_reward/tree/main/adroit) 
| MetaWorld | 7 | 140 | 38.8M | [Scripts](https://github.com/Farama-Foundation/Metaworld/blob/master/scripts/demo_sawyer.py) | [Download](https://huggingface.co/datasets/tauhuang/diffusion_reward/tree/main/metaworld)|

You can download the datasets and place them to  `/video_dataset` to reproduce the results in this paper. 

## Pretrain Reward Models
Train VQGAN encoder. 
```bash
bash scripts/run/codec_model/vqgan_${domain}.sh    # [adroit, metaworld]
```
Train video models.
```bash
bash scripts/run/video_model/${video_model}_${domain}.sh    # [vqdiffusion, videogpt]_[adroit, metaworld]
```

### (Optinal) Download Pre-trained Models
We also provide the pre-trained reward models (including Diffusion Reward and VIPER) used in this paper for result reproduction. You may download the models with configuration files [here](https://huggingface.co/tauhuang/diffusion_reward/tree/main), and place the folders in `/exp_local`.

## Train RL with Pre-trained Rewards 
Train DrQv2 with different rewards.
```bash
bash scripts/run/rl/drqv2_${domain}_${reward}.sh ${task}    # [adroit, metaworld]_[diffusion_reward, viper, viper_std, amp, rnd, raw_sparse_reward]
```
Notice that you should login [wandb](https://wandb.ai/site) for logging experiments online. Turn it off, if you aim to log locally, in configuration file [here](diffusion_reward/configs/rl//default.yaml#L24).

# 🧭 Code Navigation

```
diffusion_reward
  |- configs               # experiment configs 
  |    |- models           # configs of codec models and video models
  |    |- rl               # configs of rl 
  |
  |- envs                  # envrionments, wrappers, env maker
  |    |- adroit.py        # Adroit env
  |    |- metaworld.py     # MetaWorld env
  |    |- wrapper.py       # env wrapper and utils
  |
  |- models                # implements core codec models and video models
  |    |- codec_models     # image encoder, e.g., VQGAN
  |    |- video_models     # video prediction models, e.g., VQDiffusion and VideoGPT
  |    |- reward_models    # reward models, e.g., Diffusion Reward and VIPER
  |
  |- rl                    # implements core rl algorithms
```

# ✉️ Contact
For any questions, please feel free to email taou.cs13@gmail.com or luccachiang@gmail.com.


# 🙏 Acknowledgement
Our code is built upon [VQGAN](https://github.com/dome272/VQGAN-pytorch), [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion), [VIPER](https://github.com/Alescontrela/viper_rl), [AMP](https://github.com/med-air/DEX), [RND](https://github.com/jcwleo/random-network-distillation-pytorch), and [DrQv2](https://github.com/facebookresearch/drqv2). We thank all these authors for their nicely open sourced code and their great contributions to the community.

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

# 📝 Citation

If you find our work useful, please consider citing:
```
@article{Huang2023DiffusionReward,
  title={Diffusion Reward: Learning Rewards via Conditional Video Diffusion},
  author={Tao Huang and Guangqi Jiang and Yanjie Ze and Huazhe Xu},
  journal={arxiv}, 
  year={2023},
}
```
