#!/user/bin/python
# coding=utf-8

from dataloaders.ColorChecker import ColorCheckerLoader, default_input_transform
from models.ColorConstancyNCRF import ColorConstancyNCRF
from torch.utils.data import DataLoader
from constant import *
from utils.StatisticalValue import StatisticalValue
from utils.functions.status import print_training_status
from loss_functions.angular_loss import angular_loss
from env import writer
from torchvision import transforms
from loss_functions.cross_entropy_loss_ct import cross_entropy_loss
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

model = ColorConstancyNCRF()
model.to(device=DEVICE)

mse_loss = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


def save_training_process(root, images, labels, names, predictions):
    images, labels, predictions = images.cpu().detach().numpy().transpose(
        0, 2, 3, 1), labels.cpu().detach().numpy(), predictions.cpu().detach().numpy()

    predictions = predictions[:, np.newaxis, np.newaxis, :]
    labels = labels[:, np.newaxis, np.newaxis, :]

    ct_images = images[:, :, :, :] / predictions
    gt_images = images[:, :, :, :] / labels

    for idx, (ct_image) in enumerate(ct_images):
        combined_images = torch.zeros(3, 3, *ct_images.shape[1:3])
        combined_images[0] = torch.from_numpy(
            ct_images[idx].transpose(2, 0, 1))
        combined_images[1] = torch.from_numpy(
            gt_images[idx].transpose(2, 0, 1))
        combined_images[2] = torch.from_numpy(images[idx].transpose(2, 0, 1))

        if os.path.isdir(root) is False:
            os.makedirs(root)

        torchvision.utils.save_image(
            combined_images, os.path.join(root, names[idx].replace('tiff', 'jpg')))


def run(epoch):
    statistical_losses = StatisticalValue()
    statistical_angular_errors = StatisticalValue()

    optimizer.zero_grad()

    for idx, (images, edges, labels, names) in enumerate(fold1_train_loader):
        images, labels, edges = images.to(
            DEVICE), labels.to(DEVICE), edges.to(DEVICE)

        illuminatios, boundaries = model(images)
        loss = (mse_loss(illuminatios, labels) + cross_entropy_loss(boundaries, edges)) / ITERATION_SIZE
        loss.backward()

        with torch.no_grad():
            angular_error = angular_loss(illuminatios, labels)

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

        if (idx + 1) % 30 == 0:
            save_training_process(os.path.join(
                TMP_ROOT, 'epoch-%s-training-view' % epoch), images, labels, names, illuminatios)

    scheduler.step()

    return statistical_losses
