#!/user/bin/python
# coding=utf-8

from dataloaders.saliency_detection.DUTS import DUTS
from models.RACNN.model import RACNN as Model
from models.RACNN.loss import saliency_loss
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
        coordinate=False,
    ),
    batch_size=10,
    shuffle=False,
    num_workers=8
)

model = Model()
model.to(device=DEVICE)

state_dict = torch.load(
    '/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_unet_saliency_detection/__tmp__/2020-12-21-21-34-50/checkpoints/checkpoint_153_1.pth')['state_dict']
new_state_dict = {}
for key in state_dict:
    new_state_dict[key.replace('unet.', '')] = state_dict[key]
model.unet.load_state_dict(new_state_dict)

state_dict = torch.load('/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_detection/__tmp__/2021-01-11-14-35-41/checkpoints/checkpoint_5_1.pth')['state_dict']
model.rcnn.rcnn.load_state_dict(
    state_dict
)

criterion = saliency_loss
mae = torch.nn.L1Loss()

optimizer = torch.optim.Adam(
    model.gate.parameters(),
    lr=0.0004,
    weight_decay=0
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

def run(epoch):
    statistical_losses = StatisticalValue()
    statistical_mae_errors = StatisticalValue()
    statistical_mae_errors1 = StatisticalValue()
    statistical_mae_errors2 = StatisticalValue()

    optimizer.zero_grad()

    for idx, (images, labels, names) in enumerate(trainloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs, _1, _2 = model(images)
        outputs_list = [outputs]

        loss = saliency_loss(outputs_list, labels)
        loss.backward()

        mae_error = mae(input=outputs, target=labels)
        statistical_mae_errors.update(mae_error.item(), names)
        statistical_mae_errors1.update(
            mae(input=_1, target=labels).item(), names
        )
        statistical_mae_errors2.update(
            mae(input=_2, target=labels).item(), names
        )
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
                'Final MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_mae_errors))
            print(
                '1 MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_mae_errors1))
            print(
                '2 MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_mae_errors2))

    # scheduler.step()
    return statistical_losses
