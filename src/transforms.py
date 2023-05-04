import random
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms.v2 import functional as F


def pad_if_smaller(img, size, fill=0):
    img = img.squeeze(0)
    min_size = min(img.size())
    if min_size < size:
        ow, oh = img.size()
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img.unsqueeze(0)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, [size, size])
        target = F.resize(target, [size, size], interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class NumpyToTensor:
    def __call__(self, image, target):
        image = torch.from_numpy(image)
        target = torch.from_numpy(target).long()
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        image = F.adjust_brightness(image, self.brightness)
        image = F.adjust_contrast(image, self.contrast)
        image = F.adjust_saturation(image, self.saturation)
        image = F.adjust_hue(image, self.hue)
        return image, target


class GaussianBlur:
    def __init__(self, kernel_size, sigma, p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.gaussian_blur(image, self.kernel_size, self.sigma)
            target = F.gaussian_blur(target, self.kernel_size, self.sigma)
        return image, target


class RandomRotation:
    def __init__(self, degrees, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            degrees = random.uniform(-self.degrees, self.degrees)
            image = F.rotate(image, degrees)
            target = F.rotate(target, degrees)
        return image, target


class RandomAffine:
    def __init__(self, degrees, translate=None, scale=None, shear=None, p=0.5):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            scale = 1 - self.scale if random.random() < self.p else 1 + self.scale  # either shrink or enlarge
            shear_x = self.shear[0] if random.random() < self.p else -self.shear[0]  # either left or right
            shear_y = self.shear[1] if random.random() < self.p else -self.shear[1]  # either up or down
            translate_x = self.translate[0] if random.random() < self.p else -self.translate[0]  # either left or right
            translate_y = self.translate[1] if random.random() < self.p else -self.translate[1]  # either up or down
            shear = [shear_x, shear_y]
            translate = [translate_x, translate_y]
            image = F.affine(image,
                             angle=self.degrees,
                             translate=translate,
                             scale=scale,
                             shear=shear)
            target = F.affine(target,
                              angle=self.degrees,
                              translate=translate,
                              scale=scale,
                              shear=shear)
        return image, target


class Lambda:
    def __init__(self, lambd):
        self.lambd = lambd

    def __call__(self, image, target):
        return image, self.lambd(target)
