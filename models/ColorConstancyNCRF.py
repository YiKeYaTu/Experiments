import torch
import torch.nn as nn
from models.RCF import RCF
from constant import *


# class ColorConstancyNCRF(nn.Module):
#     def __init__(self):
#         super(ColorConstancyNCRF, self).__init__()

#         self.conv_crf = nn.Conv2d(3, 256, 7, padding=3)
#         self.conv_ncrf = nn.Conv2d(256, 256, 21, padding=10, bias=False)
#         self.conv_mf = nn.Conv2d(256, 3, 1, padding=0)
#         self.conv_glavg = nn.AdaptiveAvgPool2d(1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         conv_crf = self.conv_crf(x)
#         conv_ncrf = self.conv_ncrf(conv_crf)
#         conv_modulatory = self.relu(conv_crf - conv_ncrf)
#         conv_fusion = self.conv_mf(conv_modulatory)

#         return torch.squeeze(self.conv_glavg(conv_fusion))


class ColorConstancyNCRF(nn.Module):
    def __init__(self):
        super(ColorConstancyNCRF, self).__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        self.conv_crf = nn.Conv2d(3, 64, 7, padding=3)
        self.conv_ncrf = nn.Conv2d(64, 64, 21, padding=10, bias=False)
        # Illu estimation units;
        self.conv_illumination1_1 = nn.Conv2d(64, 128, 3)
        self.conv_illumination1_2 = nn.Conv2d(128, 128, 3)

        self.conv_illumination2_1 = nn.Conv2d(128, 256, 3)
        self.conv_illumination2_2 = nn.Conv2d(256, 256, 3)
        self.conv_illumination2_3 = nn.Conv2d(256, 256, 3)

        self.conv_illumination_fusion = nn.Conv2d(256, 3, 1)
        self.conv_glavg = nn.AdaptiveAvgPool2d(1)
        # Boundary detection units.
        self.conv_boundary1_1 = nn.Conv2d(64, 128, 3)
        self.conv_boundary1_2 = nn.Conv2d(128, 128, 3)

        self.conv_boundary2_1 = nn.Conv2d(128, 256, 3)
        self.conv_boundary2_2 = nn.Conv2d(256, 256, 3)
        self.conv_boundary2_3 = nn.Conv2d(256, 256, 3)

        self.conv_boundary_fusion = nn.Conv2d(256, 1, 1)
        self.interpolate = nn.functional.interpolate
        # Fusion units for combining boundary and illu.
        # self.conv_boundary_illumination_fusion1 = nn.Conv2d(256, 128, 1)
        # self.conv_boundary_illumination_fusion2 = nn.Conv2d(512, 256, 1)

    def forward(self, x):
        conv_crf = self.conv_crf(x)
        conv_ncrf = self.conv_ncrf(conv_crf)
        conv_modulatory = self.relu(conv_crf - conv_ncrf)
        pool_modulatory = self.maxpool(conv_modulatory)
        # Boundary stream.
        conv_boundary1_1 = self.conv_boundary1_1(pool_modulatory)
        conv_boundary1_2 = self.conv_boundary1_2(conv_boundary1_1)
        pool_boundary1 = self.maxpool(conv_boundary1_2)

        conv_boundary2_1 = self.conv_boundary2_1(pool_boundary1)
        conv_boundary2_2 = self.conv_boundary2_2(conv_boundary2_1)
        conv_boundary2_3 = self.conv_boundary2_3(conv_boundary2_2)

        conv_boundary_fusion = self.sigmoid(self.conv_boundary_fusion(conv_boundary2_3))
        interpolated_boundary_fusion = self.interpolate(
            conv_boundary_fusion, size=(x.shape[2], x.shape[3]), mode='bilinear'
        )

        # Illu stream.
        conv_illumination1_1 = self.conv_illumination1_1(pool_modulatory)
        conv_illumination1_2 = self.conv_illumination1_2(conv_illumination1_1)

        conv_boundary_illumination_fusion1 = conv_illumination1_2 * \
            self.sigmoid(conv_boundary1_2)

        pool_illumination1 = self.maxpool(conv_boundary_illumination_fusion1)

        conv_illumination2_1 = self.conv_illumination2_1(pool_illumination1)
        conv_illumination2_2 = self.conv_illumination2_2(conv_illumination2_1)
        conv_illumination2_3 = self.conv_illumination2_3(conv_illumination2_2)

        conv_boundary_illumination_fusion2 = conv_illumination2_3 * \
            self.sigmoid(conv_boundary2_3)

        conv_illumination_fusion = self.conv_illumination_fusion(
            conv_boundary_illumination_fusion2)
        vec_illumination_fusion = self.flatten(
            self.conv_glavg(conv_illumination_fusion))

        return (
            vec_illumination_fusion,
            interpolated_boundary_fusion
        )
