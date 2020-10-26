from dataloaders.ColorChecker import ColorCheckerLoader
from models.ColorConstancyNCRF import ColorConstancyNCRF
from torch.utils.data import DataLoader
from torchvision import transforms
from constant import *
from PIL import Image
from utils.StatisticalValue import StatisticalValue
import torch
import numpy as np

fold1_train_loader = DataLoader(ColorCheckerLoader(
    fold_number=1, is_training=True), batch_size=2, shuffle=True, num_workers=8)
fold1_test_loader = DataLoader(ColorCheckerLoader(
    fold_number=1, is_training=False), batch_size=10, shuffle=True, num_workers=8)

fold2_train_loader = DataLoader(ColorCheckerLoader(
    fold_number=2, is_training=True), batch_size=10, shuffle=True, num_workers=8)
fold2_test_loader = DataLoader(ColorCheckerLoader(
    fold_number=2, is_training=False), batch_size=10, shuffle=True, num_workers=8)

fold3_train_loader = DataLoader(ColorCheckerLoader(
    fold_number=3, is_training=True), batch_size=10, shuffle=True, num_workers=8)
fold3_test_loader = DataLoader(ColorCheckerLoader(
    fold_number=3, is_training=False), batch_size=10, shuffle=True, num_workers=8)

model = ColorConstancyNCRF()
model.to(device=DEVICE)

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(
    (parameter for parameter in model.parameters()
     if parameter.requires_grad is True),
    lr=LEARNING_RATE, momentum=0.9)


def run(epoch):
    statisticalLosses = StatisticalValue()
    for idx, (images, labels, names) in enumerate(fold1_train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        illus = model(images)
        loss = criterion(illus, labels) / ITERATION_SIZE
        loss.backward()

        statisticalLosses.update(loss.item())

        if (idx + 1) % ITERATION_SIZE == 0:
            optimizer.zero_grad()
            optimizer.step()
