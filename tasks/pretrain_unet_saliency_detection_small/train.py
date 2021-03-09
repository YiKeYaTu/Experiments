#!/user/bin/python
# coding=utf-8

from dataloaders.saliency_detection.DUTS import DUTS
from models.RACNN.small_model import RACNN as Model
from models.RACNN.loss import saliency_loss
from models.PFAN.loss import EdgeSaliencyLoss
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

# state_dict = torch.load(
#     '/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_unet_saliency_detection/__tmp__/2020-12-21-21-34-50/checkpoints/checkpoint_153_1.pth',
#     map_location=DEVICE
# )['state_dict']
# new_state_dict = {}
# for key in state_dict:
#     new_state_dict[key.replace('unet.', '')] = state_dict[key]
# model.unet.load_state_dict(new_state_dict)

state_dict = torch.load(
    '/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_detection/__tmp__/2021-01-11-14-35-41/checkpoints/checkpoint_17_1.pth',
    map_location=DEVICE
)['state_dict']
model.rcnn.rcnn.load_state_dict(state_dict)

# criterion = saliency_loss
criterion = EdgeSaliencyLoss(device=DEVICE)
mae = torch.nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=4 * 1e-4,
    weight_decay=0
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

def run(epoch):
    statistical_losses = StatisticalValue()
    statistical_mae_errors = StatisticalValue()

    optimizer.zero_grad()

    for idx, (images, labels, names) in enumerate(trainloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs, crops = model(images)

        # xmin, ymin, xmax, ymax = crops[0]['box']
        # view_data = torch.zeros((4, *images.shape[1:]))
        # view_data[0, :, :, :] = images[0]
        # view_data[1, :, :, :] = labels[0]
        # view_data[2, 0:3, ymin: ymax, xmin: xmax] = 1
        # view_data[3, :, :, :] = outputs[0]

        # if not os.path.isdir(os.path.join(TMP_ROOT, 'test')):
        #     os.makedirs(os.path.join(TMP_ROOT, 'test'))

        # torchvision.utils.save_image(
        #     view_data,
        #     os.path.join(TMP_ROOT, 'test/%s' % (names[0]))
        # )

        loss = torch.tensor(0.0).to(DEVICE)

        for i, (output) in enumerate(outputs):
            xmin, ymin, xmax, ymax = crops[i]['box']
            output_list = [output[0]]
            reshaped_label = labels[i][:, ymin: ymax, xmin: xmax]
            total = torch.numel(reshaped_label)
            postive = torch.count_nonzero(reshaped_label)
            negative = total - postive

            # if postive > negative:
            #     loss += saliency_loss(output_list, reshaped_label, weight_0=postive * 2 / total, weight_1=1.0)
            # else:
            #     loss += saliency_loss(output_list, reshaped_label, weight_0=1.0, weight_1=negative * 2 / total)
            # print(output[0].unsqueeze(0).shape, reshaped_label.unsqueeze(0).shape)
            if postive > negative:
                loss += criterion(output[0].unsqueeze(0), reshaped_label.unsqueeze(0), weight_0=postive * 2 / total, weight_1=1.0)
            else:
                loss += criterion(output[0].unsqueeze(0), reshaped_label.unsqueeze(0), weight_0=1.0, weight_1=negative * 2 / total)

        loss /= len(outputs)
        loss.backward()

        mae_error = torch.tensor(0.0).to(DEVICE)

        for i, (output) in enumerate(outputs):
            xmin, ymin, xmax, ymax = crops[i]['box']
            mae_error += mae(input=output[0], target=labels[i][:, ymin: ymax, xmin: xmax])
        
        mae_error /= len(outputs)

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
