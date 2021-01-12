import torch
import torch.nn as nn
from constant import *


class BaseReceptiveBlock(nn.Module):
    def __init__(self, in_channel, out_channel, maxpool=True, residual=True):
        super(BaseReceptiveBlock, self).__init__()

        self.classical = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.Conv2d(out_channel, out_channel, 3, padding=1)
        )

        _out_channel = 128 \
            if out_channel > 128 else out_channel

        self.nonclassical = nn.Sequential(
            nn.Conv2d(out_channel, _out_channel, 5, padding=2, bias=False),
            nn.Conv2d(_out_channel, _out_channel, 5, padding=2, bias=False),
            nn.Conv2d(_out_channel, out_channel, 5, padding=2, bias=False),
        )

        self.fusion = nn.Conv2d(out_channel * 2, out_channel, 1)
        self.projection = (lambda x: torch.repeat_interleave(x, int(out_channel / in_channel), dim=1)) \
            if residual is True else None
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) \
            if maxpool is True else None
        self.relu = nn.ReLU()

    def forward(self, x):
        classical = self.classical(x)
        nonclassical = self.nonclassical(classical)

        fusion = self.fusion(torch.cat((classical, nonclassical), dim=1))
        fusion = nonclassical

        if self.projection:
            projection = self.projection(x)
            fusion = fusion + projection

        if self.maxpool:
            maxpool = self.maxpool(fusion)
            return self.relu(maxpool)

        return self.relu(fusion)


class StackedNCRF(nn.Module):
    def __init__(self):
        super(StackedNCRF, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            BaseReceptiveBlock(64, 64, residual=False),
            BaseReceptiveBlock(64, 128),
            BaseReceptiveBlock(128, 256, maxpool=False),
            BaseReceptiveBlock(256, 512, maxpool=False),
            BaseReceptiveBlock(512, 512, maxpool=False),
        )

        self.confidence = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid(),
        )
        self.illumap = nn.Conv2d(512, 3, 1)
        self.relu = nn.ReLU()
        self.upsample = nn.functional.interpolate

    def forward(self, x):
        _, _, h, w = x.size()

        x = self.features(x)
        confidence = self.confidence(x)
        illumap = self.relu(self.illumap(x))

        x = illumap * confidence
        x = self.upsample(x, size=(h, w), mode='bilinear')

        return [x]
