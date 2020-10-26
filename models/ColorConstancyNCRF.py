import torch
import torch.nn as nn
from models.RCF import RCF
from constant import *


class ColorConstancyNCRF(nn.Module):
    def __init__(self):
        super(ColorConstancyNCRF, self).__init__()

        self.conv_crf = nn.Conv2d(3, 64, 7, padding=3)
        self.conv_ncrf = nn.Conv2d(64, 64, 21, padding=10, bias=False)
        self.conv_mf = nn.Conv2d(64, 3, 1, padding=0)
        self.conv_glavg = torch.mean

        self.rcf = RCF()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(18, 3)

        for para in self.rcf.parameters():
            para.requires_grad = False

        self.relu = torch.relu

    def forward(self, x):
        conv_crf = self.conv_crf(x)
        conv_ncrf = self.conv_ncrf(conv_crf)
        conv_modulatory = self.relu(conv_crf - conv_ncrf)
        # conv_contours = self.rcf(x)
        conv_fusion = self.conv_mf(conv_modulatory)

        # fusion_illu = conv_fusion * conv_contours
        # conv_glavg = self.conv_glavg(fusion_illu.view(
        #     *fusion_illu.shape[0:3], -1), dim=3).transpose(0, 1)
        # fc = self.fc(self.flatten(conv_glavg))

        # return fc

        return self.conv_glavg(conv_fusion.view(
            conv_fusion.size(0), conv_fusion.size(1), -1), dim=2)
