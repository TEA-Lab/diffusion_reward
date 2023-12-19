import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, path, size=None, is_train=True):
        self.size = size

        path = str(Path(__file__).parents[4]) + path
        self.images = []
        for root, subdirs, files in os.walk(path):
            for name in files:
                if is_train and 'train' in root:
                    self.images.append(os.path.join(root, name))
                if not is_train and 'test' in root:
                    self.images.append(os.path.join(root, name))
                
        self._length = len(self.images)
        self.preprocessor = lambda x: x

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=args.image_size, is_train=True)
    eval_data = ImagePaths(args.dataset_path, size=args.image_size, is_train=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=True)
    return train_loader, eval_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# def plot_images(images):
#     x = images["input"]
#     reconstruction = images["rec"]
#     half_sample = images["half_sample"]
#     full_sample = images["full_sample"]

#     fig, axarr = plt.subplots(1, 4)
#     axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
#     axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
#     axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
#     axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
#     plt.show()


