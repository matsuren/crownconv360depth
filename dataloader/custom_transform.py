import cv2
import numpy as np
import torch
from imgaug import augmenters as iaa
from torchvision import transforms


# -------For FisheyeToIcoDataset ---------------
class ToTensor(object):
    def __init__(self):
        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.depth = 'idepth'
        self.ToTensor = transforms.ToTensor()
        self.depToTensor = lambda x: torch.from_numpy(x).float()

    def __call__(self, sample):
        if self.depth in sample:
            sample[self.depth] = self.depToTensor(sample[self.depth][np.newaxis])
        for cam in self.cam_list:
            sample[cam] = self.ToTensor(sample[cam][np.newaxis])
        return sample


class Normalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.normalizer = transforms.Normalize(mean, std)

    def __call__(self, sample):
        for cam in self.cam_list:
            sample[cam] = self.normalizer(sample[cam])
        return sample


# -------For OmniStereoDataset ---------------
class ColorJitter(object):
    def __init__(self):
        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.some_aug = iaa.SomeOf(
            (0, 2),
            [
                iaa.AdditiveGaussianNoise(
                    loc=0,
                    scale=(0.0,
                           0.01 * 255)),  # add gaussian noise to images
                iaa.contrast.LinearContrast(
                    (0.8, 1.2),
                    per_channel=0.25),  # improve or worsen the contrast
                iaa.Multiply((0.8, 1.2), per_channel=0.25),
                iaa.Add((-25, 25), per_channel=0.25)
            ],
            random_order=True)

    def __call__(self, sample):
        for cam in self.cam_list:
            sample[cam] = self.some_aug.augment_image(sample[cam])
        return sample


class RandomShift(object):
    def __init__(self, scale=0.5):
        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.scale = scale

    def __call__(self, sample):
        for cam in self.cam_list:
            rows, cols = sample[cam].shape[:2]
            shift_x = np.random.normal(loc=0.0, scale=self.scale)
            shift_y = np.random.normal(loc=0.0, scale=self.scale)
            M = np.array([
                [1, 0, shift_x],
                [0, 1, shift_y]])
            sample[cam] = cv2.warpAffine(sample[cam], M, (cols, rows))
        return sample
