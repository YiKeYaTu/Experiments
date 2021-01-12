import torchvision

from torch.utils.data import Dataset
from constant import DATASETS_ROOT
from os import path, listdir
from PIL import Image
import numpy as np

LAB_DATASET_ROOT_PATH = path.join(
    DATASETS_ROOT, '多光源颜色恒常性数据集/多光源/benchmark_mirf/lab/img')
LAB_GROUND_TRUTH_PATH = path.join(
    DATASETS_ROOT, '多光源颜色恒常性数据集/多光源/benchmark_mirf/lab/groundtruth')
REAL_DATASET_ROOT_PATH = path.join(
    DATASETS_ROOT, '多光源颜色恒常性数据集/多光源/benchmark_mirf/realworld/img')
REAL_GROUND_TRUTH_PATH = path.join(
    DATASETS_ROOT, '多光源颜色恒常性数据集/多光源/benchmark_mirf/realworld/groundtruth')

TRAIN_SET = [
    0, 1, 2, 4, 5, 7, 8, 10, 12, 13, 15, 17, 18, 19, 20,
    21, 22, 24, 26, 27, 29, 30, 32, 33, 34, 37, 38, 39, 40,
    42, 43, 45, 46, 47, 48, 50, 53, 54, 55, 57
]


class MIRF(Dataset):
    def __init__(
        self,
        transform=None,
        train=True,
        lab=True,
        target_transform=None
    ):
        self.transform = transform
        self.train = train
        self.lab = lab
        self.target_transform = target_transform

        self.image_paths = self._get_image_paths(
            LAB_DATASET_ROOT_PATH
            if self.lab else REAL_DATASET_ROOT_PATH
        )
        self.target_paths = [
            path.join(
                LAB_GROUND_TRUTH_PATH if self.lab else REAL_DATASET_ROOT_PATH,
                p.split('/')[-1]
            ) for p in self.image_paths
        ]

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        target = Image.open(self.target_paths[index])
        
        assert self.image_paths[index].split(
            '/')[-1] == self.target_paths[index].split('/')[-1]
        assert image.size == target.size

        image = self.transform(image) \
            if self.transform is not None else image
        target = self.target_transform(target) \
            if self.target_transform is not None else target

        return image, target, self.image_paths[index].split('/')[-1]

    def __len__(self):
        return len(self.image_paths)

    def _get_image_paths(self, root):
        if self.train:
            return [
                path.join(root, p)
                for idx, (p) in enumerate(sorted(listdir(root)))
                if idx in TRAIN_SET and p[0] != '.'
            ]
        else:
            return [
                path.join(root, p)
                for idx, (p) in enumerate(sorted(listdir(root)))
                if idx not in TRAIN_SET and p[0] != '.'
            ]
