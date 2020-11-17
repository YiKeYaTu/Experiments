#!/user/bin/python
# coding=utf-8

from dataloaders.ColorChecker import ColorCheckerLoader
from models.ColorConstancyNCRF import ColorConstancyNCRF
from torch.utils.data import DataLoader
from constant import *
from PIL import Image
from utils.Logger import Logger
from utils.StatisticalValue import StatisticalValue
from loss_functions.angular_loss import angular_loss
import torch
import numpy as np
import torchvision
import sys

logger = Logger(join(TMP_ROOT, 'test.txt'))
sys.stdout = logger

fold1_test_loader = DataLoader(
    ColorCheckerLoader(
        fold_number=1,
        is_training=False,
        input_transform=torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(1024),
            torchvision.transforms.ToTensor(),
        ]) 
    ),
    batch_size=1,
    num_workers=1,
    shuffle=False
)


with torch.no_grad():
    model = ColorConstancyNCRF()
    model.to(device=DEVICE)

def run():
    statistical_angular_errors = StatisticalValue()
    
    with torch.no_grad():
        for idx, (images, labels, names) in enumerate(fold1_test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # print(images.shape)

            illus = model(images)

            angular_error = angular_loss(illus, labels)
            statistical_angular_errors.update(angular_error.item(), names)

            images, illus = images.cpu().detach().numpy().transpose(
                0, 2, 3, 1), illus.cpu().detach().numpy()
            illus = illus[:, np.newaxis, np.newaxis, :]

            images = images[:, :, :, :] / illus

            # print(images.shape, illus.shape)

            # exit()

                # result = Image.fromarray((images[0]).astype(np.uint8))
                # result.save(join(TMP_ROOT, "corrected_%s.png") % names[0])

            print(
                'Angular Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}' \
                    .format(errors=statistical_angular_errors)
            )

            # print("Running test [%d/%d]" % (idx + 1, len(fold1_test_loader)))
