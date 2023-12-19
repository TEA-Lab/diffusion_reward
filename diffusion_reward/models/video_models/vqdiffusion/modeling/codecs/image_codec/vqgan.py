import math
from pathlib import Path

import torch
from diffusion_reward.models.codec_models.vqgan.vqgan import VQGAN


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


class MiniVQGAN(VQGAN):
    def __init__(
        self,
        args,
        token_shape=None,
        trainable=False,
        ckpt_path=None,
        latent_size=16
    ):
        args = AttrDict(args)
        super(VQGAN, self).__init__()

        ckpt_path = str(Path(__file__).parents[7]) + ckpt_path

        self.model = VQGAN(args)
        self.model.load_checkpoint(ckpt_path)
        self.model.eval()

        self.token_shape = token_shape

    def preprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range 0-255
        """
        imgs = imgs.div(127.5) - 1  # map to -1 - 1
        return imgs
        # return map_pixels(imgs)   
    
    def postprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range -1 - 1
        """
        imgs = (imgs + 1) * 127.5
        return imgs.clip(0, 255)

    def get_tokens(self, imgs):
        if imgs.max() >= 3: 
            imgs = self.preprocess(imgs)
        if imgs.dim() == 4:
            embs, code, _ = self.model.encode(imgs)
            #output = {'token': code.reshape([embs.shape[0], self.token_shape[0], self.token_shape[1]])}
            output = {'token': code.reshape([embs.shape[0], -1])}
        elif imgs.dim() == 5:
            # serve as cond tokens, no dict
            flat_imgs = imgs.flatten(0, 1)
            embs, code, _ = self.model.encode(flat_imgs)
            output = code.reshape([imgs.shape[0], -1])
        return output

    @torch.no_grad()
    def encode_to_z(self, x):
        if x.max() >= 3: 
            x = self.preprocess(x)
        if len(x.shape) == 5:
            flat_x = x.flatten(0, 1)
            quant_z, indices, _ = self.model.encode(flat_x)
        else:
            quant_z, indices, _ = self.model.encode(x)

        indices = indices.reshape(x.shape[0], -1)
        #indices = indices.view(quant_z.shape[0], -1)
        quant_z = quant_z.permute(0, 2, 3, 1)
        quant_z = quant_z.reshape(x.shape[0], -1, quant_z.shape[-1])
        return quant_z, indices

    def decode(self, z):
        latent_size = int(math.sqrt(z.shape[1]))
        assert latent_size ** 2 == z.shape[1]
        #z = z.reshape([z.shape[0], latent_size, latent_size])
        
        ix_to_vectors = self.model.codebook.embedding(z).reshape([z.shape[0], latent_size, latent_size, self.model.codebook.latent_dim])
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.model.decode(ix_to_vectors)
        return self.postprocess(image)
