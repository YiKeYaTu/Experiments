import torch
import torch.nn as nn
from models.RCF import RCF
from constant import *


class ColorConstancyNCRF(nn.Module):
    def __init__(self):
        super(ColorConstancyNCRF, self).__init__()

        self.conv_crf = nn.Conv2d(3, 256, 7, padding=3)
        self.conv_ncrf = nn.Conv2d(256, 256, 21, padding=10, bias=False)
        self.conv_mf = nn.Conv2d(256, 3, 1, padding=0)
        self.conv_glavg = torch.mean
        self.relu = torch.relu

    def forward(self, x):
        conv_crf = self.conv_crf(x)
        conv_ncrf = self.conv_ncrf(conv_crf)
        conv_modulatory = self.relu(conv_crf - conv_ncrf)
        conv_fusion = self.conv_mf(conv_modulatory)

        return self.conv_glavg(conv_fusion.view(
            conv_fusion.size(0), conv_fusion.size(1), -1), dim=2)
