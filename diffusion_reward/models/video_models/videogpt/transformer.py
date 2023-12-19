from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...codec_models.vqgan.vqgan import VQGAN
from .mingpt import GPT


class VideoGPTTransformer(nn.Module):
    def __init__(self, args):
        super(VideoGPTTransformer, self).__init__()

        self.vqgan = self.load_vqgan(args)
        self.transformer = GPT(**args.transformer)

        self.sos_token = args.sos_token
        self.pkeep = args.pkeep
        self.args = args
        self.use_vqemb = args.use_vqemb

    @staticmethod
    def load_vqgan(args):
        args.codec.checkpoint_path = str(Path(__file__).parents[4]) + args.codec.checkpoint_path
        model = VQGAN(args.codec)
        model.load_checkpoint(args.codec.checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        if len(x.shape) == 5:
            flat_x = x.flatten(0, 1)
            quant_z, indices, _ = self.vqgan.encode(flat_x)
        else:
            quant_z, indices, _ = self.vqgan.encode(x)

        indices = indices.reshape(x.shape[0], -1)
        quant_z = quant_z.permute(0, 2, 3, 1)
        quant_z = quant_z.reshape(x.shape[0], -1, quant_z.shape[-1])
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices):
        p = self.args.latent_size
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0] * (indices.shape[1] // p // p), p, p, self.args.code_dim)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)

        image = self.vqgan.decode(ix_to_vectors)
        return image

    def calc_sos_tokens(self, x, embs):
        if not self.use_vqemb:
            sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
            sos_tokens = sos_tokens.long().to("cuda")
        else:
            sos_tokens = torch.ones(x.shape[0], embs.shape[-1]) * self.sos_token
            sos_tokens = sos_tokens.long().to("cuda")[:, None, :]
        return sos_tokens

    def output(self, x, compute_joint=True):
        embs, indices = self.encode_to_z(x)
        target = indices
        sos_tokens = self.calc_sos_tokens(x, embs)

        if not self.use_vqemb:
            # here we feed indices to transformer
            mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
            mask = mask.round().to(dtype=torch.int64)
            random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
            new_indices = mask * indices + (1 - mask) * random_indices

            new_indices = torch.cat((sos_tokens, indices), dim=1)
            logits, _ = self.transformer(new_indices[:, :-1])
        else:
            # here we feed embs to transformer
            mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
            mask = mask.round().to(dtype=torch.int64)[:, :, None]
            new_embs = embs * mask
            new_embs = torch.cat([sos_tokens, new_embs], dim=1)

            logits, _ = self.transformer(new_embs[:, :-1])

        if compute_joint:
            return logits, target
        else:
            num_valid_logits = int(logits.shape[1] / x.shape[1])
            return logits[:, -num_valid_logits:], target[:, -num_valid_logits:]

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, embs, x_, c, steps, temperature=1.0, top_k=1):
        self.transformer.eval()
        if not self.use_vqemb:
            x = torch.cat((c, x_), dim=1) if x_ is not None else c    # prior sampling
        else:
            x = torch.cat((c, embs), dim=1) if x_ is not None else c
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")
        indices = torch.cat((sos_tokens, x_), dim=1) if x_ is not None else c

        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)
            indices = torch.cat((indices, ix), dim=1)
            if self.use_vqemb:
                ix = self.vqgan.codebook.embedding(ix).reshape(ix.shape[0], 1, -1)

            x = torch.cat((x, ix), dim=1)

        indices = indices[:, c.shape[1]:]
        self.transformer.train()
        return indices

    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        embs, indices = self.encode_to_z(x)
        sos_tokens = self.calc_sos_tokens(x, embs)

        start_embs = embs[:, :-indices.shape[1] // x.shape[1], :]
        start_indices = indices[:, :-indices.shape[1] // x.shape[1]]
        
        sample_indices = self.sample(start_embs, start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1])
        
        half_sample = self.z_to_image(sample_indices)

        log["input"] = x[0, :, :, :, :]
        log["half_sample"] = half_sample[-1].unsqueeze(0)
        return log, torch.concat((x[0, :, :, :, :], half_sample[-1].unsqueeze(0)))


