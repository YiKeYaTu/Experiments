from constant import DATASETS_ROOT
from torch.utils.data import Dataset
from os import listdir, path
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import scipy.io as scio
import sys
import os
import re

DATASET_ROOT_PATH = path.join(
    DATASETS_ROOT, 'ColorChecker_Recommended')
GROUND_TRUTH_PATH = path.join(
    DATASETS_ROOT, 'groundtruth&coordinates/ColorCheckerData.mat')
MASK_ROOT_PATH = path.join(
    DATASETS_ROOT, 'masks'
)
FOLDS_PATH = path.join(
    DATASETS_ROOT, 'ColorChecker_Recommended_Folds.mat')
INPUT_SHAPE = (512, 512)


def to_numpy(x):
    return np.array(x, dtype=np.float32)


def transpose(image):
    return image.transpose(2, 0, 1)


def input_transform():
    return transforms.Compose([
        transforms.RandomCrop(INPUT_SHAPE),
        # transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(0.5),
        to_numpy,
        transpose,
    ])
    return None


def target_transform():
    return transforms.Compose([
        to_numpy,
    ])


class ColorCheckerLoader(Dataset):
    def __init__(self, fold_number, is_training=True, input_transform=input_transform, target_transform=target_transform):
        self.is_training = is_training
        self.folds = scio.loadmat(FOLDS_PATH)[
            'tr_split' if self.is_training is True else 'te_split'][0]
        self.fold_number = fold_number - 1

        self.illuminants = self.__filter__(
            scio.loadmat(GROUND_TRUTH_PATH)['REC_groundtruth'])
        self.image_names = self.__filter__(sorted(
            [image for idx, (image) in enumerate(listdir(
                DATASET_ROOT_PATH)) if image[0] != '.'],
            key=lambda name: int(name.split('_')[0])))

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __add_mask__(self, image_name, image):
        mask = np.array(Image.open(
            path.join(MASK_ROOT_PATH, 'mask1_' + re.sub(r'^\d+_', '', image_name))))
        image[mask == 0] = 0
        return transforms.ToPILImage()(image)

    def __filter__(self, data):
        return [d for idx, (d) in enumerate(data) if (idx + 1) in self.folds[self.fold_number]]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = np.array(Image.open(
            path.join(DATASET_ROOT_PATH, self.image_names[index])))
        # Covering the standard color board
        image = self.__add_mask__(self.image_names[index], image)
        illuminant = self.illuminants[index]

        if self.is_training is True:
            if self.input_transform is not None:
                image = self.input_transform()(image)

            if self.target_transform is not None:
                illuminant = self.target_transform()(illuminant)

        return image, illuminant, self.image_names[index]
