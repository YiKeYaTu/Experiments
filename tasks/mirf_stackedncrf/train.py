#!/user/bin/python
# coding=utf-8

from dataloaders.multi_color_constancy.MIRF import MIRF
from models.multi_color_constancy.StackedNCRF import StackedNCRF
from torch.utils.data import DataLoader
from constant import DEVICE, LEARNING_RATE, ITERATION_SIZE, WEIGHT_DECAY, TMP_ROOT
from utils.StatisticalValue import StatisticalValue
from utils.functions.status import print_training_status
from loss_functions.multi_angular_loss import multi_angular_loss
from env import iteration_writer
from torchvision import transforms
from os.path import join
import torch
import torchvision
import numpy as np
import os

trainloader = DataLoader(
    MIRF(
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        train=True,
        lab=True,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8
)

model = StackedNCRF()
model.to(device=DEVICE)
model.load_state_dict(
    torch.load('/home/ncrc-super/data/Liangchen/Experiments/tasks/color_constancy_multi/__tmp__/2020-11-19-15-29-24/checkpoints/checkpoint_27_1.pth')['state_dict']
)

criterion = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

def run(epoch):
    statistical_losses = StatisticalValue()
    statistical_angular_errors = StatisticalValue()

    optimizer.zero_grad()

    for idx, (images, labels, names) in enumerate(trainloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        predictions = model(images)

        loss = torch.zeros(1).to(DEVICE)

        for p in predictions:
            loss += criterion(p, labels) / ITERATION_SIZE

        loss.backward()

        angular_error = multi_angular_loss(predictions[-1], labels)
        statistical_angular_errors.update(angular_error.item(), names)
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
                'Angular Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_angular_errors))

    scheduler.step()
    return statistical_losses
