import torch
import torch.nn as nn
from constant import *
# model refers to
# 2020-11-19-15-29-24
class BaseReceptiveBlock(nn.Module):
    def __init__(self, in_channel, out_channel, maxpool=True, residual=True):
        super(BaseReceptiveBlock, self).__init__()

        self.classical = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.Conv2d(out_channel, out_channel, 3, padding=1)
        )

        self.nonclassical = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, bias=False),
            nn.Conv2d(out_channel, out_channel, 5, padding=2, bias=False),
            nn.Conv2d(out_channel, out_channel, 5, padding=2, bias=False),
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
            BaseReceptiveBlock(3, 64, residual=False),
            BaseReceptiveBlock(64, 128),
            BaseReceptiveBlock(128, 256),
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

        return [ x ]

# model refers to
# 2020-11-21-11-24-23
# class BaseReceptiveBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, maxpool=True, residual=True):
#         super(BaseReceptiveBlock, self).__init__()

#         self.classical = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, 3, padding=1),
#             nn.Conv2d(out_channel, out_channel, 3, padding=1)
#         )

#         self.nonclassical = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, 5, padding=2, bias=False),
#             nn.Conv2d(out_channel, out_channel, 5, padding=2, bias=False),
#             nn.Conv2d(out_channel, out_channel, 5, padding=2, bias=False),
#         )

#         self.fusion = nn.Conv2d(out_channel * 2, out_channel, 1)
#         self.projection = (lambda x: torch.repeat_interleave(x, int(out_channel / in_channel), dim=1)) \
#             if residual is True else None
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) \
#             if maxpool is True else None
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         classical = self.classical(x)
#         nonclassical = self.nonclassical(classical)
#         fusion = self.fusion(torch.cat((classical, nonclassical), dim=1))

#         if self.projection:
#             projection = self.projection(x)
#             fusion = fusion + projection

#         if self.maxpool:
#             maxpool = self.maxpool(fusion)
#             return self.relu(maxpool)

#         return self.relu(fusion)


# class MultiColorConstancy(nn.Module):
#     def __init__(self):
#         super(MultiColorConstancy, self).__init__()

#         self.stag1 = BaseReceptiveBlock(3, 64, residual=False)
#         self.stag2 = BaseReceptiveBlock(64, 128)
#         self.stag3 = BaseReceptiveBlock(128, 256)
#         self.stag4 = BaseReceptiveBlock(256, 512, maxpool=False)
#         self.stag5 = BaseReceptiveBlock(512, 512, maxpool=False)

#         self.illumap_stage1 = nn.Conv2d(64, 3, 1)
#         self.illumap_stage2 = nn.Conv2d(128, 3, 1)
#         self.illumap_stage3 = nn.Conv2d(256, 3, 1)
#         self.illumap_stage4 = nn.Conv2d(512, 3, 1)
#         self.illumap_stage5 = nn.Conv2d(512, 3, 1)
#         self.illumap_fusion = nn.Conv2d(3 * 5, 3, 1)

#         self.relu = nn.ReLU()
#         self.upsample = nn.functional.interpolate

#     def forward(self, x):
#         stag1 = self.stag1(x)
#         stag2 = self.stag2(stag1)
#         stag3 = self.stag3(stag2)
#         stag4 = self.stag4(stag3)
#         stag5 = self.stag5(stag4)

#         illumap_stage1 = self.illumap_stage1(stag1)
#         illumap_stage2 = self.illumap_stage2(stag2)
#         illumap_stage3 = self.illumap_stage3(stag3)
#         illumap_stage4 = self.illumap_stage4(stag4)
#         illumap_stage5 = self.illumap_stage5(stag5)

#         up_illumap_stage1 = self.upsample(
#             illumap_stage1, scale_factor=2, mode='bilinear')
#         up_illumap_stage2 = self.upsample(
#             illumap_stage2, scale_factor=4, mode='bilinear')
#         up_illumap_stage3 = self.upsample(
#             illumap_stage3, scale_factor=8, mode='bilinear')
#         up_illumap_stage4 = self.upsample(
#             illumap_stage4, scale_factor=8, mode='bilinear')
#         up_illumap_stage5 = self.upsample(
#             illumap_stage5, scale_factor=8, mode='bilinear')

#         illumap_fusion = self.illumap_fusion(
#             torch.cat(
#                 (
#                     up_illumap_stage1,
#                     up_illumap_stage2,
#                     up_illumap_stage3,
#                     up_illumap_stage4,
#                     up_illumap_stage5
#                 ), dim=1
#             )
#         )

#         return [up_illumap_stage1, up_illumap_stage2, up_illumap_stage3, up_illumap_stage4, up_illumap_stage5, illumap_fusion]
