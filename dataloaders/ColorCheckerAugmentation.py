from constant import DATASETS_ROOT
from torch.utils.data import Dataset
from os import listdir, path
from utils.functions.rotate_image import *
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import os

def _gen_aug_images():
    angles = np.arange(0, 360, 20)
    flips = [1]

    image_size = (512, 512)

    image_names = [image for idx, (image) in enumerate(listdir(
        DATASET_ROOT_PATH)) if image[0] != '.']
    saving_image_names = []

    for image_name in image_names:
        image_file = cv2.imread(
            path.join(DATASET_ROOT_PATH, image_name))
        image_file = add_mask(image_name, image_file)
        image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)

        H, W, C = image_file.shape

        for angle in angles:
            for flip in flips:
                cur_x = 0
                while cur_x + image_size[0] <= W:
                    cur_y = 0
                    while cur_y + image_size[1] <= H:
                        ident = '%.2f_%d_%d_%d' % (angle, flip, cur_x, cur_y)
                        saving_root = os.path.join(
                            DATASETS_ROOT, 'ColorChecker_Augmentation', ident)

                        if os.path.isdir(saving_root) is False:
                            os.makedirs(saving_root)

                        # if os.path.isfile(os.path.join(saving_root, image_name)) is True:
                        #     print('%s already exist.' % os.path.join(saving_root, image_name))
                        #     continue

                        aug_image_file = transforms.functional.to_pil_image(
                            image_file)
                        aug_image_file = transforms.functional.crop(
                            aug_image_file, cur_y, cur_x, image_size[0], image_size[1])
                        if flip == 1:
                            aug_image_file = transforms.functional.hflip(
                                aug_image_file)
                        if angle > 0:
                            aug_image_file = np.array(aug_image_file)
                            aug_image_file = rotate_image(
                                aug_image_file, angle)
                            aug_image_file = crop_around_center(
                                aug_image_file,
                                *largest_rotated_rect(
                                    image_size[0],
                                    image_size[1],
                                    math.radians(angle)
                                )
                            )
                            aug_image_file = transforms.functional.to_pil_image(
                                aug_image_file)
                        aug_image_file.save(
                            os.path.join(saving_root, image_name))
                        saving_image_names.append(ident + '/' + image_name)
                        print(os.path.join(saving_root, image_name))

                        cur_y += image_size[1]
                        if cur_y < H and cur_y + image_size[1] > H:
                            cur_y = H - image_size[1]

                    cur_x += image_size[0]
                    if cur_x < W and cur_x + image_size[0] > W:
                        cur_x = W - image_size[0]

    file = open(os.path.join(DATASETS_ROOT,
                             'ColorChecker_Augmentation', 'data.lst'), 'a')
    file.write('\n'.join(sorted(sorted(saving_image_names,
                                       key=lambda name: int(name.split('/')[1].split('_')[0])), key=lambda name: name.split('/')[0])))
