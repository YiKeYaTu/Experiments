import torch
import torch.nn as nn
import torchvision
import os
from models.RCF import RCF
from constant import *


class ColorConstancyNCRF(nn.Module):
    def __init__(self):
        super(ColorConstancyNCRF, self).__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.conv_crf = nn.Conv2d(3, 256, 7, padding=3)
        self.conv_ncrf = nn.Conv2d(256, 256, 21, padding=10, bias=False)
        # Illu estimation units;
        self.conv_illumination1_1 = nn.Conv2d(256, 256, 3)
        self.conv_illumination1_2 = nn.Conv2d(256, 256, 3)

        self.conv_illumination2_1 = nn.Conv2d(256, 512, 3)
        self.conv_illumination2_2 = nn.Conv2d(512, 512, 3)
        self.conv_illumination2_3 = nn.Conv2d(512, 512, 3)

        self.conv_illumination_fusion = nn.Conv2d(512, 3, 1)
        self.conv_glavg = nn.AdaptiveAvgPool2d(1)
        # Boundary detection units.
        # self.conv_boundary1_1 = nn.Conv2d(256, 256, 3)
        # self.conv_boundary1_2 = nn.Conv2d(256, 256, 3)

        # self.conv_boundary2_1 = nn.Conv2d(256, 512, 3)
        # self.conv_boundary2_2 = nn.Conv2d(512, 512, 3)
        # self.conv_boundary2_3 = nn.Conv2d(512, 512, 3)

        # self.conv_boundary_fusion = nn.Conv2d(512, 1, 1)
        # self.interpolate = nn.functional.interpolate

    def forward(self, x):
        conv_crf = self.conv_crf(x)
        conv_ncrf = self.conv_ncrf(conv_crf)
        conv_modulatory = self.relu(conv_crf - conv_ncrf)
        pool_modulatory = self.maxpool(conv_modulatory)
        # Illu stream.
        conv_illumination1_1 = self.conv_illumination1_1(pool_modulatory)
        conv_illumination1_2 = self.conv_illumination1_2(conv_illumination1_1)
        pool_illumination1 = self.maxpool(conv_illumination1_2)

        conv_illumination2_1 = self.conv_illumination2_1(pool_illumination1)
        conv_illumination2_2 = self.conv_illumination2_2(conv_illumination2_1)
        conv_illumination2_3 = self.conv_illumination2_3(conv_illumination2_2)
        pool_illumination2 = self.maxpool(conv_illumination2_3)

        conv_illumination_fusion = self.conv_illumination_fusion(pool_illumination2)
        vec_illumination_fusion = self.flatten(self.conv_glavg(conv_illumination_fusion))
        # Boundary stream.
        # conv_boundary1_1 = self.conv_boundary1_1(pool_modulatory)
        # conv_boundary1_2 = self.conv_boundary1_2(conv_boundary1_1)
        # pool_boundary1 = self.maxpool(conv_boundary1_2)

        # conv_boundary2_1 = self.conv_boundary2_1(pool_boundary1)
        # conv_boundary2_2 = self.conv_boundary2_2(conv_boundary2_1)
        # conv_boundary2_3 = self.conv_boundary2_3(conv_boundary2_2)
        # pool_boundary2 = self.maxpool(conv_boundary2_3)

        # conv_boundary_fusion = self.conv_boundary_fusion(pool_boundary2)
        # interpolated_boundary_fusion = self.interpolate(conv_boundary_fusion, size=(x.shape[2], x.shape[3], mode='bilinear'))

        return vec_illumination_fusion
