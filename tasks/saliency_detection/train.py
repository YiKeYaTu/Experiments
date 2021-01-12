#!/user/bin/python
# coding=utf-8

from dataloaders.saliency_detection.DUTS import DUTS
from models.saliency_detection.VGG16 import VGG16
from torch.utils.data import DataLoader
from constant import DEVICE, LEARNING_RATE, ITERATION_SIZE, WEIGHT_DECAY, TMP_ROOT
from utils.StatisticalValue import StatisticalValue
from utils.functions.status import print_training_status
from env import iteration_writer
from torchvision import transforms
from os.path import join
import torch
import torchvision
import numpy as np
import os

trainloader = DataLoader(
    DUTS(
        train=True,
        transform=transforms.Compose([
            transforms.CenterCrop(size=(256, 256)),
            transforms.ToTensor(),
        ]),
        target_transform=transforms.Compose([
            transforms.CenterCrop(size=(256, 256)),
            transforms.ToTensor(),
        ]),
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8
)

model = VGG16()
model.to(device=DEVICE)

pos_weight = torch.empty((256, 256))
pos_weight[:, :] = 1.12

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
mae = torch.nn.L1Loss()

# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=1e-2,
#     momentum=0.9,
#     weight_decay=0
# )
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=0
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

def run(epoch):
    statistical_losses = StatisticalValue()
    statistical_mae_errors = StatisticalValue()

    optimizer.zero_grad()

    for idx, (images, labels, names) in enumerate(trainloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # images = images * 255

        predictions = model(images)

        loss = criterion(input=predictions, target=labels) / ITERATION_SIZE

        loss.backward()

        mae_error = mae(input=torch.nn.functional.sigmoid(predictions), target=labels)
        statistical_mae_errors.update(mae_error.item(), names)
        statistical_losses.update(loss.item())

        iteration_writer.add_scalar(
            'Loss/Iteration',
            statistical_losses.val[0],
            (epoch - 1) * len(trainloader) + idx + 1
        )

        if (idx + 1) % ITERATION_SIZE == 0:
            optimizer.step()
            optimizer.zero_grad()

            print_training_status(epoch, idx + 1, len(trainloader),
                                  statistical_losses.val[0], statistical_losses.avg)
            print(
                'MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_mae_errors))

    # scheduler.step()

    return statistical_losses
