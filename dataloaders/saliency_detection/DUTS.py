import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
from torch.utils.data import Dataset

if __name__ == '__main__':
    import sys
    import os
    abspath = os.path.dirname(__file__)
    sys.path.append(os.path.join(abspath, '../../'))

from constant import DATASETS_ROOT
from utils.ImagePairReader import ImagePairReader
from utils.functions.image_pair import *

TR_IMG_PATH = path.join(
    DATASETS_ROOT, 'saliency_deteciton/DUTS-TR/DUTS-TR-Image')
TR_GT_PATH = path.join(
    DATASETS_ROOT, 'saliency_deteciton/DUTS-TR/DUTS-TR-Mask')
TE_IMG_PATH = path.join(
    DATASETS_ROOT, 'saliency_deteciton/DUTS-TE/DUTS-TE-Image')
TE_GT_PATH = path.join(
    DATASETS_ROOT, 'saliency_deteciton/DUTS-TE/DUTS-TE-Mask')

def cal_center(img):
    img = img.squeeze()
    count = torch.sum(img)

    x = torch.stack([torch.arange(0, img.size(1))] * img.size(0)).float()
    y = torch.stack([torch.arange(0, img.size(0))] * img.size(1)).t().float()

    x = torch.sum(x * img) / count
    y = torch.sum(y * img) / count

    x = int(x)
    y = int(y)

    return (x, y)

def cal_coordinate(img):
    tx, ty = cal_center(img)

    left_most = 1e9
    right_most = -1
    top_most = 1e9
    bottom_most = -1

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                if i < top_most:
                    top_most = i
                if i > bottom_most:
                    bottom_most = i
                if j < left_most:
                    left_most = j
                if j > right_most:
                    right_most = j

    tr = right_most - tx
    tb = bottom_most - ty
    tl = tx - left_most
    tt = ty - top_most

    return left_most, top_most, right_most, bottom_most

    # return tx, ty, tr, tb, tl, tt

class DUTS(Dataset):
    def __init__(
        self,
        train=True,
        augment=True,
        coordinate=True,
        transform=None,
        target_transform=None,
        target_size=256
    ):
        self.image_pair_reader = ImagePairReader(
            TR_IMG_PATH if train else TE_IMG_PATH,
            TR_GT_PATH if train else TE_GT_PATH
        )

        self.train = train
        self.augment = train and augment
        self.coordinate = coordinate
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.transform = transform
        self.target_transform = target_transform
        self.target_size = target_size

    def __getitem__(self, idx):
        name, img, gt = self.image_pair_reader.get_item(idx)

        if self.augment:
            img, gt = random_crop_flip(img, gt)
            img, gt = random_rotate(img, gt)
            img = random_brightness(img)

        if self.target_size:
            img, gt = pad_resize_image(img, gt, self.target_size)

        img = np.transpose(img, axes=(2, 0, 1))
        img = torch.from_numpy(img / 255.0).float()
        img = self.normalize(img)

        if self.coordinate:
            coordinate = torch.tensor(cal_coordinate(torch.from_numpy(gt))).float()
            coordinate = torch.unsqueeze(coordinate, dim=0)

        # tx, ty, tr, tb, tl, tt = cal_coordinate(torch.from_numpy(gt))

        # plt.subplot(121)
        # plt.imshow(gt, cmap='gray')
        # plt.subplot(122)
        # gt[ty - tt:ty + tb, tx - tl:tx + tr] = 255
        # plt.imshow(gt, cmap='gray')
        # plt.show()

        gt = np.expand_dims(gt, axis=0)
        gt = torch.from_numpy(gt / np.max(gt)).float()

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            gt = self.target_transform(gt)

        if self.coordinate:
            return (
                img,
                gt,
                coordinate,
                name
            )
        else:
            return (
                img,
                gt,
                name
            )

    def __len__(self):
        return self.image_pair_reader.get_len()

if __name__ == '__main__':
    dataset = DUTS()
    dataset.__getitem__(1)