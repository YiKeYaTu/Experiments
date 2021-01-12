import torch
import torch.nn as nn
import torchvision
import os
from models.RCF import RCF
from constant import *


class ColorConstancyWithAlexNet(nn.Module):
    def __init__(self):
        super(ColorConstancyWithAlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 64, kernel_size=6, padding=0),
            nn.Conv2d(64, 3, kernel_size=1, padding=0),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.squeeze(x)

