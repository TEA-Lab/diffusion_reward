import os
import shutil
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from diffusion_reward.models.codec_models.vqgan.discriminator import Discriminator
from diffusion_reward.models.codec_models.vqgan.lpips import LPIPS
from diffusion_reward.models.codec_models.vqgan.utils import load_data, weights_init
from diffusion_reward.models.codec_models.vqgan.vqgan import VQGAN
from torchvision import utils as vutils
from tqdm import tqdm


class TrainVQGAN:
    def __init__(self, args):
        self.args = args
        self.work_dir = 'results'
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    def prepare_training(self):
        if os.path.exists(f"{self.work_dir}/visualization"):
            shutil.rmtree(f"{self.work_dir}/visualization")
        os.makedirs(f"{self.work_dir}/visualization", exist_ok=True)
        if os.path.exists(f"{self.work_dir}/checkpoints"):
            shutil.rmtree(f"{self.work_dir}/checkpoints") 
        os.makedirs(f"{self.work_dir}/checkpoints", exist_ok=True)

    def train(self, args):
        train_dataset, val_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        best_loss = float('inf')
        for epoch in range(args.epochs):
            os.makedirs(f"{self.work_dir}/visualization/epoch{epoch}", exist_ok=True)
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % 100 == 0:
                        with torch.no_grad():
                            #print(decoded_images.min(), decoded_images.max(), 'here')
                            real_fake_images = torch.cat((imgs.add(1).mul(0.5)[:4], decoded_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join(f"{self.work_dir}/visualization/epoch{epoch}", f"{i}.jpg"), nrow=4)

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)
                val_loss = self.eval(val_dataset, epoch*steps_per_epoch+i)
                is_best = val_loss < best_loss
                best_loss = min(best_loss, val_loss)
                if is_best:
                    torch.save(self.vqgan.state_dict(), os.path.join(f"{self.work_dir}/checkpoints", f"vqgan.pt"))
                    print(f'Checkpoint at epoch {epoch} is saved with eval loss {best_loss} !!!')

    def eval(self, val_dataset, step):
        losses = []
        for imgs in val_dataset:
            imgs = imgs.to(device=self.args.device)
            decoded_images, _, q_loss = self.vqgan(imgs)

            disc_fake = self.discriminator(decoded_images)
            disc_factor = self.vqgan.adopt_weight(self.args.disc_factor, step, threshold=self.args.disc_start)

            perceptual_loss = self.perceptual_loss(imgs, decoded_images)
            rec_loss = torch.abs(imgs - decoded_images)
            perceptual_rec_loss = self.args.perceptual_loss_factor * perceptual_loss + self.args.rec_loss_factor * rec_loss
            perceptual_rec_loss = perceptual_rec_loss.mean()
            g_loss = -torch.mean(disc_fake)

            λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
            vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss
            losses.append(vq_loss.cpu().detach().numpy().item())
        loss = np.array(losses).mean()
        return loss

@hydra.main(config_path="../diffusion_reward/configs/models/codec_models/vqgan", config_name="default")
def main(cfg):
    TrainVQGAN(cfg)


if __name__ == '__main__':
    main()