#!/user/bin/python
# coding=utf-8

from dataloaders.saliency_detection.DUTS import DUTS
from models.PFAN.recurrent_model import RecurrentPFAN as Model
from models.PFAN.loss import EdgeSaliencyLoss
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
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8
)

model = Model()
model.to(device=DEVICE)

criterion = EdgeSaliencyLoss(device=DEVICE)
mae = torch.nn.L1Loss()

def find_center(mask):
    center_x = 0
    center_y = 0
    center_count = 0
    center_mask = torch.zeros_like(mask)

    for x in range(mask.shape[2]):
        for y in range(mask.shape[3]):
            if mask[0][0][x][y] > 0.8:
                center_x += x * mask[0][0][x][y]
                center_y += y * mask[0][0][x][y]
                center_count += 1
                
    center = (center_x / center_count, center_y / center_count)
    center = (int(torch.ceil(center[0]).item()), int(torch.ceil(center[1]).item()))
    center_mask[0][0][center[0]][center[1]] = 255

    return center_mask

def run():
    statistical_losses = StatisticalValue()
    statistical_mae_errors = StatisticalValue()
    sub_dir = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print('Test start.')

    with torch.no_grad():

        for idx, (images, labels, names) in enumerate(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs, intermediate_images = model(images)
            pred_masks, ca_act_regs = outputs[0]
            # center_mask = find_center(pred_masks)

            loss = (criterion(pred_masks, labels) + ca_act_regs) / ITERATION_SIZE

            mae_error = mae(input=pred_masks, target=labels)
            statistical_mae_errors.update(mae_error.item(), names)
            statistical_losses.update(loss.item())

            iteration_writer.add_scalar(
                'Loss/Test',
                statistical_losses.val[0],
                idx + 1,
            )

            if idx % 1 == 0:
                view_data = torch.zeros((4, *images.shape[1:]))
                view_data[0, :, :, :] = images.squeeze()
                view_data[1, :, :, :] = pred_masks.squeeze()
                view_data[2, :, :, :] = labels.squeeze()
                # view_data[3, :, :, :] = center_mask.squeeze()

                if not os.path.isdir(os.path.join(TMP_ROOT, 'test', sub_dir)):
                    os.makedirs(os.path.join(TMP_ROOT, 'test', sub_dir))

                torchvision.utils.save_image(
                    view_data,
                    os.path.join(TMP_ROOT, 'test/%s/%s' % (sub_dir, names[0]))
                )

            print(
                'mae Error: mean: {errors.avg}, mid: {errors.mid}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_mae_errors))
            break

        return statistical_losses
