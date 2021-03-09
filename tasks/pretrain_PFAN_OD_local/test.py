#!/user/bin/python
# coding=utf-8

from dataloaders.saliency_detection.DUTS import DUTS
from models.SOD.PFAN_OD.model import PFAN_OD
from models.PFAN.loss import EdgeSaliencyLoss
from torch.utils.data import DataLoader
from constant import DEVICE, LEARNING_RATE, WEIGHT_DECAY, TMP_ROOT
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
        augment=True,
        coordinate=False
    ),
    batch_size=5,
    shuffle=False,
    num_workers=8
)
testloader = DataLoader(
    DUTS(
        train=False,
        augment=False,
        coordinate=False,
    ),
    batch_size=10,
    shuffle=False,
    num_workers=8
)

model = PFAN_OD(mode='train_local')
model.to(device=DEVICE)

criterion = EdgeSaliencyLoss(device=DEVICE)
l1_loss = torch.nn.L1Loss()

def run():
    statistical_losses = StatisticalValue()
    statistical_mae_errors = StatisticalValue()
    statistical_mae_errors1 = StatisticalValue()
    statistical_mae_errors2 = StatisticalValue()

    print('Test start.')

    with torch.no_grad():
        for idx, (images, labels, names) in enumerate(testloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs, global_result, local_result, crops = model(images)
            _1 = local_result[0]

            # statistical_mae_errors.update(mae(input=outputs, target=labels).item(), names)
            statistical_mae_errors1.update(l1_loss(input=_1, target=labels).item(), names)
            # statistical_mae_errors2.update(l1_loss(input=_2, target=labels).item(), names)

            # print(
            #     'Final MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
            #         errors=statistical_mae_errors))
            print(
                'Index: {idx}, Local MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_mae_errors1, idx=idx))
            # print(
            #     'Global MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
            #         errors=statistical_mae_errors2))

            for iidx in range(len(images)):
                xmin, ymin, xmax, ymax = crops[iidx]['box']
                view_data = torch.zeros((4, *images.shape[1:]))
                view_data[0, :, :, :] = images[iidx]
                view_data[1, :, :, :] = labels[iidx]
                view_data[2, 0:3, ymin: ymax, xmin: xmax] = 1
                view_data[3, :, :, :] = _1[iidx]
                # view_data[4, :, :, :] = _2[iidx]

                if not os.path.isdir(os.path.join(TMP_ROOT, 'test')):
                    os.makedirs(os.path.join(TMP_ROOT, 'test'))

                torchvision.utils.save_image(
                    view_data,
                    os.path.join(TMP_ROOT, 'test/%s' % (names[0]))
                )
                break

    print('Test End.')