# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import numpy as np
import torchvision.transforms.functional as tf


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class Jigsaw(object):
    """Split an image into n*n patches, randomly permute them and then stack them together to get a new image of
    the original size
    """

    def __init__(self, n=2, perm=None):
        """
        Args:
            n (int): size. the image will be split into n*n patches
            perm (tuple): the permutation according to which the new image is stacked. If perm[i] = j, it means to
                            move the patch at i//n row and i%n column to j//n row and j%n column
        """
        self.n = n
        self.perm = perm

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): the input image
        """
        (w, h) = img.size
        sub_w = w // self.n
        sub_h = h // self.n
        if self.perm is None:
            self.perm = np.random.permutation(self.n*self.n)

        # new_img = Image.new('RGB', (sub_w * self.n, sub_h * self.n))
        # for i in range(self.n*self.n):
        #     # get the tile at i//n row and i%n column
        #     row = i // self.n
        #     col = i % self.n
        #     sub_img = img.crop((col * sub_w, row * sub_h, (col+1) * sub_w, (row+1) * sub_h))
        #     # paste the tile into the new image
        #     new_row = self.perm[i] // self.n
        #     new_col = self.perm[i] % self.n
        #     new_img.paste(sub_img, (new_col * sub_w, new_row * sub_h))
        img_cpy = np.copy(np.array(img))
        new_img = np.zeros_like(img_cpy)
        for i in range(self.n*self.n):
            row = i // self.n
            col = i % self.n
            new_row = self.perm[i] // self.n
            new_col = self.perm[i] % self.n
            new_img[new_col * sub_w : (new_col+1) * sub_w, new_row * sub_h : (new_row+1) * sub_h] = img_cpy[col * sub_w : (col+1) * sub_w, row * sub_h : (row+1) * sub_h]
        new_img = Image.fromarray(new_img)

        new_img.resize((w, h))
        return new_img