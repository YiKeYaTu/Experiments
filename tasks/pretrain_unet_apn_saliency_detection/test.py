#!/user/bin/python
# coding=utf-8

from dataloaders.saliency_detection.DUTS import DUTS
from models.RACNN.model import RACNN as Model
from models.RACNN.loss import loc_loss as criterion
from torch.utils.data import DataLoader
from constant import DEVICE, LEARNING_RATE, ITERATION_SIZE, WEIGHT_DECAY, TMP_ROOT
from utils.StatisticalValue import StatisticalValue
from utils.functions.status import print_training_status
from env import iteration_writer
from torchvision import transforms
from os.path import join
import time
import torch
import torchvision
import numpy as np
import os

trainloader = DataLoader(
    DUTS(
        train=False,
        augment=False,
        coordinate=True,
    ),
    batch_size=10,
    shuffle=False,
    num_workers=8
)

model = Model()
model.to(device=DEVICE)

# criterion = saliency_loss
mae = torch.nn.L1Loss()

def run():
    statistical_losses = StatisticalValue()
    statistical_mae_errors = StatisticalValue()
    sub_dir = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print('Test start.')

    with torch.no_grad():

        for idx, (images, labels, coordinates, names) in enumerate(trainloader):
            images, coordinates = images.to(DEVICE), coordinates.to(DEVICE)

            _, locs = model(images)

            loss = criterion([locs], [coordinates]) / ITERATION_SIZE

            # mae_error = mae(input=outputs, target=labels)
            # statistical_mae_errors.update(mae_error.item(), names)
            statistical_losses.update(loss.item())

            iteration_writer.add_scalar(
                'Loss/Test',
                statistical_losses.val[0],
                idx + 1,
            )

            if idx % 1 == 0:
                print(
                    'mae Error: mean: {errors.avg}, mid: {errors.mid}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                        errors=statistical_losses))

        return statistical_losses
