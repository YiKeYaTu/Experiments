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
    batch_size=6,
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

model = PFAN_OD(mode='train_global')
model.to(device=DEVICE)

criterion = EdgeSaliencyLoss(device=DEVICE)
l1_loss = torch.nn.L1Loss()

optimizer = None
# optimizer_local = torch.optim.Adam(
#     model.local_net.parameters(),
#     lr=0.0004,
#     weight_decay=0
# )

optimizer_global = torch.optim.Adam(
    model.global_net.parameters(),
    lr=0.0004,
    weight_decay=0
)
def test():
    statistical_losses = StatisticalValue()
    statistical_mae_errors = StatisticalValue()
    statistical_mae_errors1 = StatisticalValue()
    statistical_mae_errors2 = StatisticalValue()

    print('Test start.')

    with torch.no_grad():
        for idx, (images, labels, names) in enumerate(testloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs, global_result, local_result, crops = model(images)
            _2 = global_result[0]

            # statistical_mae_errors.update(mae(input=outputs, target=labels).item(), names)
            # statistical_mae_errors1.update(l1_loss(input=_1, target=labels).item(), names)
            statistical_mae_errors2.update(l1_loss(input=_2, target=labels).item(), names)

            # print(
            #     'Final MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
            #         errors=statistical_mae_errors))
            # print(
            #     'Local MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
            #         errors=statistical_mae_errors1))
            print(
                'Global MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_mae_errors2))

            for iidx in range(len(images)):
                # xmin, ymin, xmax, ymax = crops[iidx]['box']
                view_data = torch.zeros((5, *images.shape[1:]))
                view_data[0, :, :, :] = images[iidx]
                view_data[1, :, :, :] = labels[iidx]
                # view_data[2, 0:3, ymin: ymax, xmin: xmax] = 1
                # view_data[3, :, :, :] = _1[iidx]
                view_data[4, :, :, :] = _2[iidx]

                if not os.path.isdir(os.path.join(TMP_ROOT, 'test')):
                    os.makedirs(os.path.join(TMP_ROOT, 'test'))

                torchvision.utils.save_image(
                    view_data,
                    os.path.join(TMP_ROOT, 'test/%s' % (names[0]))
                )
                break

    print('Test End.')

def run(epoch):
    statistical_local_losses = StatisticalValue()
    statistical_global_losses = StatisticalValue()

    statistical_local_l1_errors = StatisticalValue()
    statistical_global_l1_errors = StatisticalValue()

    # optimizer_local.zero_grad()
    optimizer_global.zero_grad()

    for idx, (images, labels, names) in enumerate(trainloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        _, global_result, local_result, crops = model(images)

        # local_loss = torch.tensor(0.0).to(DEVICE)

        # for i, (output) in enumerate(local_result[0]):
        #     xmin, ymin, xmax, ymax = crops[i]['box']
        #     reshaped_output = output[:, ymin: ymax, xmin: xmax]
        #     reshaped_label = labels[i][:, ymin: ymax, xmin: xmax]
        #     total = torch.numel(reshaped_label)
        #     postive = torch.count_nonzero(reshaped_label)
        #     negative = total - postive

        #     reshaped_output = reshaped_output.unsqueeze(0)
        #     reshaped_label = reshaped_label.unsqueeze(0)

        #     if postive > negative:
        #         local_loss += criterion(reshaped_output, reshaped_label, weight_0=postive * 2 / total, weight_1=1.0)
        #     else:
        #         local_loss += criterion(reshaped_output, reshaped_label, weight_0=1.0, weight_1=negative * 2 / total)

        # local_loss /= len(local_result[0])
        # local_loss += local_result[1]
        global_loss = criterion(global_result[0], labels) + global_result[1]

        # local_loss.backward()
        global_loss.backward()

        # statistical_local_losses.update(local_loss.item())
        statistical_global_losses.update(global_loss.item())

        # local_l1_error = torch.tensor(0.0).to(DEVICE)

        # for i, (output) in enumerate(local_result[0]):
        #     xmin, ymin, xmax, ymax = crops[i]['box']
        #     local_l1_error += l1_loss(input=output[:, ymin: ymax, xmin: xmax], target=labels[i][:, ymin: ymax, xmin: xmax])
        
        # local_l1_error /= len(local_result[0])

        # statistical_local_l1_errors.update(local_l1_error)
        statistical_global_l1_errors.update(l1_loss(input=global_result[0], target=labels))

        if (idx + 1) % 10 == 0:
            # optimizer_local.step()
            # optimizer_local.zero_grad()

            optimizer_global.step()
            optimizer_global.zero_grad()

            # print('Local Loss: ')
            # print_training_status(epoch, idx + 1, len(trainloader),
            #                       statistical_local_losses.val[0], statistical_local_losses.avg)
            print('Global Loss: ')
            print_training_status(epoch, idx + 1, len(trainloader),
                                  statistical_global_losses.val[0], statistical_global_losses.avg)
            # print(
            #     'Local MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
            #         errors=statistical_local_l1_errors))
            print(
                'Global MAE Error: mean: {errors.avg}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_global_l1_errors))

    return statistical_global_losses
