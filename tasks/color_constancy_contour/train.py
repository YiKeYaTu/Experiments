#!/user/bin/python
# coding=utf-8

from dataloaders.ColorChecker import ColorCheckerLoader
from models.ColorConstancWithContourDetection import ColorConstancWithContourDetection
from torch.utils.data import DataLoader
from constant import *
from utils.StatisticalValue import StatisticalValue
from utils.functions.status import print_training_status
from loss_functions.angular_loss import angular_loss
from env import writer
from torchvision import transforms
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

model = ColorConstancWithContourDetection()
model.to(device=DEVICE)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


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
            combined_images,
            os.path.join(root, names[idx].replace('tiff', 'jpg'))
        )


def run(epoch):
    statistical_losses = StatisticalValue()
    statistical_angular_errors = StatisticalValue()

    optimizer.zero_grad()

    for idx, (images, edges, labels, names) in enumerate(fold1_train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        predictions = model(images)

        loss = torch.zeros(1).to(DEVICE)

        for p in predictions:
            loss += criterion(p, labels) / ITERATION_SIZE

        loss.backward()

        angular_error = angular_loss(predictions[-1], labels)
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

        # if (idx + 1) % 10 == 0:
        #     save_training_process(os.path.join(
        #         TMP_ROOT, 'epoch-%s-training-view' % epoch), images, labels, names, predictions)

    scheduler.step()

    return statistical_losses
