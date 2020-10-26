import torch
import torch.nn as nn


class Fine(nn.Module):
    def __init__(self):
        super(Fine, self).__init__()

        self.conv_CRF = nn.Conv2d(3, 256, 7, padding=3)
        self.conv_NCRF = nn.Conv2d(256, 256, 21, padding=10, bias=False)
        self.conv_MF = nn.Conv2d(256, 1, 1, padding=0)

        self.relu = torch.relu

    def forward(self, x):
        conv_CRF = self.conv_CRF(x)
        conv_NCRF = self.conv_NCRF(conv_CRF)

        conv_modulatory = self.relu(conv_CRF - conv_NCRF)

        conv_fusion = self.conv_MF(conv_modulatory)

        return conv_fusion

class Medium(nn.Module):
    def __init__(self):
        super(Medium, self).__init__()

        self.conv_CRF = nn.Conv2d(3, 256, 7, padding=3)
        self.conv_NCRF = nn.Conv2d(256, 256, 21, padding=10, bias=False)
        self.conv_MF = nn.Conv2d(256, 1, 1, padding=0)

        self.down_sample = nn.functional.interpolate
        self.up_sample = nn.functional.interpolate

        self.relu = torch.relu

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.down_sample(x, scale_factor=1/2, mode='bilinear')

        conv_CRF = self.conv_CRF(x)
        conv_NCRF = self.conv_NCRF(conv_CRF)

        conv_modulatory = self.relu(conv_CRF - conv_NCRF)

        conv_fusion = self.conv_MF(conv_modulatory)

        conv_fusion = self.up_sample(conv_fusion, size=(h, w), mode='bilinear')

        return conv_fusion

class Coarse(nn.Module):
    def __init__(self):
        super(Coarse, self).__init__()

        self.conv_CRF = nn.Conv2d(3, 256, 5, padding=2)
        self.conv_NCRF = nn.Conv2d(256, 256, 15, padding=7, bias=False)
        self.conv_MF = nn.Conv2d(256, 1, 1, padding=0)

        self.down_sample = nn.functional.interpolate
        self.up_sample = nn.functional.interpolate

        self.relu = torch.relu

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.down_sample(x, scale_factor=1/4, mode='bilinear')

        conv_CRF = self.conv_CRF(x)
        conv_NCRF = self.conv_NCRF(conv_CRF)

        conv_modulatory = self.relu(conv_CRF - conv_NCRF)

        conv_fusion = self.conv_MF(conv_modulatory)

        conv_fusion = self.up_sample(conv_fusion, size=(h, w), mode='bilinear')

        return conv_fusion
