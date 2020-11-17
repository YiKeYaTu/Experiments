from constant import DATASETS_ROOT
from torch.utils.data import Dataset
from os import listdir, path
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import scipy.io as scio
import sys
import os
import re
import random

DATASET_ROOT_PATH = path.join(
    DATASETS_ROOT, 'ColorChecker_Recommended')
GROUND_TRUTH_PATH = path.join(
    DATASETS_ROOT, 'groundtruth&coordinates/ColorCheckerData.mat')
MASK_ROOT_PATH = path.join(
    DATASETS_ROOT, 'masks'
)
EDGE_ROOT_PATH = path.join(
    DATASETS_ROOT, 'boundaries'
)
FOLDS_PATH = path.join(
    DATASETS_ROOT, 'ColorChecker_Recommended_Folds.mat')
INPUT_SHAPE = (512, 512)


def default_input_transform(image, i, j, h, w, angle=0, flip=0):
    image = transforms.functional.to_pil_image(image)
    image = transforms.functional.crop(image, i, j, h, w)
    image = transforms.functional.rotate(image, angle)
    image = transforms.functional.hflip(image) if flip == 1 else image

    return image


def add_mask(image_name, image):
    mask = cv2.imread(
        path.join(
            MASK_ROOT_PATH, 'mask1_' + re.sub(r'^\d+_', '', image_name)
        )
    )
    image[mask == 0] = 0
    return image


def get_edge_map(image_name, radius=30):
    edge_map = cv2.imread(
        path.join(
            EDGE_ROOT_PATH, 'boundary0_' + image_name
        ),
        cv2.IMREAD_UNCHANGED
    )
    edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2RGB)
    blur_edge_map = cv2.blur(edge_map, (radius, radius))

    return blur_edge_map[:, :, 0]


class ColorCheckerLoader(Dataset):
    def __init__(
        self,
        fold_number=None,
        is_training=True,
        should_load_boundaries=False,
        input_transform=None,
        target_transform=None
    ):
        self.is_training = is_training
        self.should_load_boundaries = should_load_boundaries

        self.folds = scio.loadmat(FOLDS_PATH)[
            'tr_split' if self.is_training is True else 'te_split'][0]
        self.fold_number = fold_number - 1 if fold_number is not None else None

        self.illuminants = [
            np.array(illuminant, dtype=np.float32)
            for illuminant in self._filter(scio.loadmat(GROUND_TRUTH_PATH)['REC_groundtruth'])
        ]
        self.image_names = self._filter(
            sorted(
                [image for idx, (image) in enumerate(
                    listdir(DATASET_ROOT_PATH)
                ) if image[0] != '.'],
                key=lambda name: int(name.split('_')[0])
            )
        )

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = cv2.imread(
            path.join(DATASET_ROOT_PATH,
                      self.image_names[index]),
            cv2.IMREAD_UNCHANGED
        )
        edge_map = get_edge_map(
            self.image_names[index], 20
        ) if self.should_load_boundaries is True else None
        illuminant = self.illuminants[index]

        # Covering the standard color board
        if self.is_training is True:
            image = add_mask(self.image_names[index], image)

        # Converting the bgr channel to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_training:
            i, j, h, w = transforms.RandomCrop.get_params(
                transforms.functional.to_tensor(image),
                INPUT_SHAPE
            )
            angle = random.uniform(0, 359)
            flip = 0
            if torch.rand(1) < 0.5:
                flip = 1

            image = default_input_transform(image, i, j, h, w, angle, flip)
            edge_map = default_input_transform(
                edge_map, i, j, h, w, angle, flip
            ) if edge_map is not None else None

        if self.input_transform is not None:
            image = self.input_transform(image)
            edge_map = self.input_transform(
                edge_map) if edge_map is not None else None

        if self.target_transform is not None:
            illuminant = self.target_transform(illuminant)

        if self.should_load_boundaries:
            return (
                image,
                edge_map,
                illuminant,
                self.image_names[index],
            )
        else:
            return (
                image,
                illuminant,
                self.image_names[index],
            )

    def _filter(self, data):
        if self.fold_number is not None:
            return [
                d for idx, (d) in enumerate(data)
                if (idx + 1) in self.folds[self.fold_number]
            ]
        else:
            return data


def _test():
    dataset = ColorCheckerLoader(
        fold_number=1, should_load_boundaries=True, is_training=True)
    images, edge_maps, illuminants, names = dataset.__getitem__(0)

    images.show()
    edge_maps.show()
