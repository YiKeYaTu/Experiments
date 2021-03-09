#!/user/bin/python
# coding=utf-8

from dataloaders.saliency_detection.DUTS import DUTS
from models.RACNN.small_model import RACNN as Model
from models.RACNN.loss import saliency_loss
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
        coordinate=False,
    ),
    batch_size=10,
    shuffle=False,
    num_workers=8
)

model = Model()
model.to(device=DEVICE)

state_dict = torch.load(
    '/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_detection/__tmp__/2021-01-11-14-35-41/checkpoints/checkpoint_17_1.pth',
    map_location=DEVICE
)['state_dict']
model.rcnn.rcnn.load_state_dict(state_dict)

criterion = saliency_loss
mae = torch.nn.L1Loss()

def run():
    statistical_mae_errors = StatisticalValue()
    sub_dir = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print('Test start.')

    with torch.no_grad():

        for idx, (images, labels, names) in enumerate(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs, crops = model(images)

            mae_error = torch.tensor(0.0).to(DEVICE)
            for i, (output) in enumerate(outputs):
                xmin, ymin, xmax, ymax = crops[i]['box']
                mae_error += mae(input=output[0], target=labels[i][:, ymin: ymax, xmin: xmax])
            
            mae_error /= len(outputs)

            statistical_mae_errors.update(mae_error.item(), names)

            if idx % 1 == 0:
                xmin, ymin, xmax, ymax = crops[0]['box']
                view_data = torch.zeros((4, *images.shape[1:]))
                view_data[0, :, :, :] = images[0]
                view_data[1, :, :, :] = labels[0]
                view_data[2, :, :, :] = torch.nn.ConstantPad2d(
                    (xmin, images[0].shape[2] - xmax,
                     ymin, images[0].shape[1] - ymax),
                    0
                )(outputs[0])
                view_data[3, 0:3, ymin: ymax, xmin: xmax] = 1

                if not os.path.isdir(os.path.join(TMP_ROOT, 'test', sub_dir)):
                    os.makedirs(os.path.join(TMP_ROOT, 'test', sub_dir))

                torchvision.utils.save_image(
                    view_data,
                    os.path.join(TMP_ROOT, 'test/%s/%s' % (sub_dir, names[0]))
                )

            print(
                'mae Error: mean: {errors.avg}, mid: {errors.mid}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_mae_errors))

