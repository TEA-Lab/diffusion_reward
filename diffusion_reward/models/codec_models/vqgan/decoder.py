import torch.nn as nn

from .helper import (GroupNorm, NonLocalBlock, ResidualBlock, Swish,
                     UpSampleBlock)


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        #channels = [512, 256, 256, 128, 128]
        channels = args.channels[::-1]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = args.latent_size

        in_channels = channels[0]
        layers = [nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            #if i != 0 and resolution < 64:
            if resolution < args.resolution:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2
        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)