#!/user/bin/python
# coding=utf-8

from dataloaders.saliency_detection.DUTS import DUTS
# from models.RACNN.model import RACNN as Model
from torchvision.models.resnet import resnet50 as Model
from models.RACNN.loss import loc_loss as criterion
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
        coordinate=True,
    ),
    batch_size=10,
    shuffle=False,
    num_workers=8
)

model = Model()
model.fc = torch.nn.Linear(in_features=2048, out_features=4, bias=True)
model.to(device=DEVICE)
print(model)

state_dict = torch.load('/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_unet_saliency_detection/__tmp__/2020-12-21-21-34-50/checkpoints/checkpoint_121_1.pth')['state_dict']
new_state_dict = {}
for key in state_dict:
    new_state_dict[key.replace('unet.', '')] = state_dict[key]

# model.unet.load_state_dict(new_state_dict)

mae = torch.nn.L1Loss()

optimizer = torch.optim.Adam(
    # model.apn.parameters(),
    model.parameters(),
    lr=0.0004,
    weight_decay=0
)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

def run(epoch):
    statistical_losses = StatisticalValue()
    statistical_mae_errors = StatisticalValue()

    optimizer.zero_grad()

    for idx, (images, _, coordinates, names) in enumerate(trainloader):
        images, coordinates = images.to(DEVICE), coordinates.to(DEVICE)
        
        locs = model(images)
        print(locs.shape, coordinates.shape)

        loss = criterion([locs], [coordinates])
        loss.backward()

        # mae_error = mae(input=outputs, target=labels)
        # statistical_mae_errors.update(mae_error.item(), names)
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
            # print(
            #     'MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
            #         errors=statistical_mae_errors))

    # scheduler.step()
        # exit()
    return statistical_losses
