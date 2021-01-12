import torch
import torch.nn as nn
from torchvision.transforms.functional import crop
from models.PFAN.model import PFAN
from constant import DEVICE
from utils.functions.boxcar import boxcar_mask

class APN(nn.Module):
    def __init__(self):
        super(APN, self).__init__()
        self.conv_1 = nn.Conv2d(128, 128, 1)
        self.conv_2 = nn.Conv2d(128, 3, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.gap(x)
        x = self.relu(x)

        '''
            (x[0], x[1]) refers to the (x, y) of center
            x[2] refers to the size of the box.
        '''

        return x

class RecurrentPFAN(nn.Module):
    def __init__(self, apn_count=1):
        super(RecurrentPFAN, self).__init__()
        self.apn_count = apn_count

        self.pfan = PFAN()
        self.apns = []
        self.interpolate = nn.functional.interpolate

        for i in range(apn_count):
            self.apns.append(APN().to(DEVICE))

    def forward(self, x):
        outputs = []
        intermediate_images = []
        _, _, h, w = x.size()

        for apn in self.apns:
            results, ca_act_reg, fused_feats = self.pfan(x)
            coordinates = apn(fused_feats).unsqueeze(3)

            outputs.append((
                results, ca_act_reg, coordinates
            ))

            x = x * boxcar_mask(x, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
            intermediate_images.append(x)

        return outputs, intermediate_images