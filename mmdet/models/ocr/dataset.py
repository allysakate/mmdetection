import os
import sys
import math
import torch

from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms


class ResizeNormalize(object):

    def __init__(self, size, interpolation = cv2.INTER_CUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = cv2.resize(img, self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

def image_transform(image, imgH=32, imgW=100, keep_ratio_with_pad=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if keep_ratio_with_pad:  # same concept with 'Rosetta' paper
        resized_max_w = imgW
        # input_channel = 3 if image.mode == 'RGB' else 1
        transform = NormalizePAD((input_channel, imgH, resized_max_w))

        w, h = image.size
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = math.ceil(imgH * ratio)
        resized_image = cv2.resize(image, (resized_w, imgH), interpolation = cv2.INTER_CUBIC)
        image_tensor = transform(resized_image)
        # resized_image.save('./image_test/%d_test.jpg' % w)
        image_tensor = image_tensor.unsqueeze(0)

    else:
        transform = ResizeNormalize((imgW, imgH))
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
