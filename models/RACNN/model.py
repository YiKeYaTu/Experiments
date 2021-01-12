import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

if __name__ == '__main__':
    import sys
    import os
    abspath = os.path.dirname(__file__)
    sys.path.append(os.path.join(abspath, '../../'))

from models.RACNN.unet_model import UNet
from constant import DEVICE
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels * 2),
            nn.Conv2d(in_channels * 2, in_channels * 2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        alphas = self._gate_conv(
            torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1))
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class RCNN(nn.Module):
    def __init__(self, **kwargs):
        super(RCNN, self).__init__()

        rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            progress=True,
            box_detections_per_img=1,
            **kwargs
        )

        rcnn.roi_heads.box_predictor = FastRCNNPredictor(
            rcnn.roi_heads.box_predictor.cls_score.in_features,
            2
        )

        self.rcnn = rcnn

    def forward(self, images):
        self.rcnn.eval()
        with torch.no_grad():
            images = list(img.to(DEVICE) for img in images)
            pred = self.rcnn(images)

            return pred


class RACNN(nn.Module):
    def __init__(self):
        super(RACNN, self).__init__()
        self.unet = UNet(3, 1)
        self.rcnn = RCNN()
        self.gate = GatedSpatialConv2d(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        rcnn_outputs = self.rcnn(x)
        crops = []

        for idx, (rcnn_output) in enumerate(rcnn_outputs):
            xmin, ymin, xmax, ymax = rcnn_output['boxes'][0]
            xmin, ymin, xmax, ymax = torch.floor(xmin).int(), torch.floor(ymin).int(), torch.floor(xmax).int(), torch.floor(ymax).int()
            # xmin, ymin, xmax, ymax = 20, 17, 200, 110
            crops.append({
                'box': (xmin, ymin, xmax, ymax),
                'image': x[idx][:, ymin: ymax, xmin: xmax]
            })

        logits1, features1 = self.unet(x)
        logits2 = []
        features2 = []

        for crop in crops:
            xmin, ymin, xmax, ymax = crop['box']
            _logits2, _features2 = self.unet(crop['image'].unsqueeze(dim=0))
            logits2.append(
                nn.ConstantPad2d(
                    (xmin, x[idx].shape[2] - xmax,
                     ymin, x[idx].shape[1] - ymax),
                    0
                )(_logits2.squeeze())
            )
            features2.append(
                nn.ConstantPad2d(
                    (xmin, x[idx].shape[2] - xmax,
                     ymin, x[idx].shape[1] - ymax),
                    0
                )(_features2.squeeze())
            )

        logits2 = torch.stack(logits2)
        features2 = torch.stack(features2)

        outputs = self.gate(features1, features2)

        return self.sigmoid(outputs), self.sigmoid(logits1), self.sigmoid(logits2)


if __name__ == '__main__':
    from models.RACNN.loss import loc_loss
    net = RACNN()
    net.to(DEVICE)
    state_dict = torch.load(
        '/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_unet_saliency_detection/__tmp__/2020-12-21-21-34-50/checkpoints/checkpoint_153_1.pth')['state_dict']
    new_state_dict = {}
    for key in state_dict:
        new_state_dict[key.replace('unet.', '')] = state_dict[key]
    net.unet.load_state_dict(new_state_dict)

    state_dict = torch.load('/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_detection/__tmp__/2021-01-11-14-35-41/checkpoints/checkpoint_5_1.pth')['state_dict']
    net.rcnn.rcnn.load_state_dict(
        state_dict
    )
    
    input = torch.randn((10, 3, 224, 224), requires_grad=True).to(DEVICE)
    ouput, _, _ = net(input)

    print(ouput.shape)

    # loss = loc_loss([locs], [torch.randn(10, 6).to(DEVICE)])
    # loss.backward()

    # print(loss)
