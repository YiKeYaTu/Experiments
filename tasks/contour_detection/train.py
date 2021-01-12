#!/user/bin/python
# coding=utf-8

from dataloaders.ColorChecker import ColorCheckerLoader
from models.ColorConstancWithContourDetection import ContourDetection
from torch.utils.data import DataLoader
from constant import *
from utils.StatisticalValue import StatisticalValue
from utils.functions.status import print_training_status
from loss_functions.angular_loss import angular_loss
from env import writer
from torchvision import transforms
import torch
import torchvision
import numpy as np
import os

fold1_train_loader = DataLoader(
    ColorCheckerLoader(
        fold_number=1, is_training=True, should_load_boundaries=True,
        input_transform=transforms.ToTensor()
    ),
    batch_size=10,
    shuffle=True,
    num_workers=8
)

model = ContourDetection()
model.to(device=DEVICE)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

def run(epoch):
    statistical_losses = StatisticalValue()
    statistical_angular_errors = StatisticalValue()

    optimizer.zero_grad()

    for idx, (images, edges, labels, names) in enumerate(fold1_train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        predictions = model(images)
        exit()

        loss = torch.zeros(1).to(DEVICE)

        for p in predictions:
            loss += criterion(p, labels) / ITERATION_SIZE

        loss.backward()

        angular_error = angular_loss(predictions[-1], labels)
        statistical_angular_errors.update(angular_error.item(), names)
        statistical_losses.update(loss.item())

        if (idx + 1) % ITERATION_SIZE == 0:
            optimizer.step()
            optimizer.zero_grad()

            print_training_status(epoch, idx + 1, len(fold1_train_loader),
                                  statistical_losses.val[0], statistical_losses.avg)
            print(
                'Angular Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_angular_errors))

    scheduler.step()

    return statistical_losses
