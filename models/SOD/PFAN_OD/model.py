import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math

if __name__ == '__main__':
    import sys
    import os
    abspath = os.path.dirname(__file__)
    sys.path.append(os.path.join(abspath, '../../../'))

from models.PFAN.model import PFAN
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

        input_features = ((input_features + gating_features) * alphas)
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class RCNN(nn.Module):
    def __init__(self, **kwargs):
        super(RCNN, self).__init__()

        with torch.no_grad():
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


class PFAN_OD(nn.Module):
    def __init__(self, mode=None):
        super(PFAN_OD, self).__init__()
        self.global_net = PFAN()
        self.local_net = PFAN()
        self.rcnn = RCNN()
        self.gate = GatedSpatialConv2d(128, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_rcnn()

        if mode == 'train_global':
            self.local_net = None
            self.rcnn = None
            self.gate = None
        elif mode == 'train_local':
            self.global_net = None
            self.gate = None
        elif mode == 'train_local_global':
            self.gate = None

    def _init_rcnn(self):
        state_dict = torch.load(
            '/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_detection/__tmp__/2021-01-11-14-35-41/checkpoints/checkpoint_17_1.pth',
            map_location=DEVICE
        )['state_dict']
        self.rcnn.rcnn.load_state_dict(state_dict)

    def forward(self, x):
        if self.rcnn:
            with torch.no_grad():
                rcnn_outputs = self.rcnn(x)
                crops = []

                for idx, (rcnn_output) in enumerate(rcnn_outputs):
                    if len(rcnn_output['boxes']) == 0:
                        xmin, ymin, xmax, ymax = 0, 0, x[idx].shape[2] - 1, x[idx].shape[1] - 1
                    else:
                        xmin, ymin, xmax, ymax = rcnn_output['boxes'][0]
                        xmin, ymin, xmax, ymax = torch.floor(xmin).int(), torch.floor(ymin).int(), torch.floor(xmax).int(), torch.floor(ymax).int()
                        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                        if xmin >= xmax or ymin >= ymax or xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
                            xmin = ymin = 0
                            xmax = x[idx].shape[2] - 1
                            ymax = x[idx].shape[1] - 1

                    padding = 10

                    xmin = xmin - padding if xmin > padding else 0
                    ymin = ymin - padding if ymin > padding else 0
                    xmax = xmax + padding if xmax < x[idx].shape[2] - padding else x[idx].shape[2] - 1
                    ymax = ymax + padding if ymax < x[idx].shape[1] - padding else x[idx].shape[1] - 1

                    width = xmax - xmin
                    height = ymax - ymin

                    padding_ho = math.ceil(32 - width)
                    padding_ve = math.ceil(32 - height)

                    if width < 32:
                        xmin -= math.ceil(padding_ho / 2)
                        xmax += math.ceil(padding_ho / 2)

                        if xmin < 0:
                            xmax -= xmin
                            xmin = 0
                        
                        elif xmax >= x[idx].shape[2]:
                            xmin -= (xmax - (x[idx].shape[2] - 1))
                            xmax = x[idx].shape[2] - 1

                    if height < 32:
                        ymin -= math.ceil(padding_ve / 2)
                        ymax += math.ceil(padding_ve / 2)

                        if ymin < 0:
                            ymax -= ymin
                            ymin = 0
                        
                        elif ymax >= x[idx].shape[1]:
                            ymin -= (ymax - (x[idx].shape[1] - 1))
                            ymax = x[idx].shape[1] - 1

                    crops.append({
                        'box': (xmin, ymin, xmax, ymax),
                        'image': x[idx][:, ymin: ymax, xmin: xmax]
                    })

        if self.global_net:
            logits1, ca_act_regs1, features1 = self.global_net(x)

        if self.local_net:
            logits2 = []
            features2 = []

            for crop in crops:
                xmin, ymin, xmax, ymax = crop['box']
                _logits2, ca_act_regs2, _features2 = self.local_net(crop['image'].unsqueeze(dim=0))
                logits2.append(
                    nn.ConstantPad2d(
                        (xmin, x[idx].shape[2] - xmax,
                        ymin, x[idx].shape[1] - ymax),
                        0
                    )(_logits2.squeeze(0))
                )
                features2.append(
                    nn.ConstantPad2d(
                        (xmin, x[idx].shape[2] - xmax,
                        ymin, x[idx].shape[1] - ymax),
                        0
                    )(_features2.squeeze(0))
                )

            logits2 = torch.stack(logits2)
            features2 = torch.stack(features2)

        if self.global_net and self.local_net:
            if self.gate:
                outputs = self.gate(features1, features2)

                outputs = self.sigmoid(outputs)

                return outputs, [logits1, ca_act_regs1], [logits2, ca_act_regs2], crops
            else:
                return None, [logits1, ca_act_regs1], [logits2, ca_act_regs2], crops
        elif self.global_net and not self.local_net:
            return None, [logits1, ca_act_regs1], None, None   
        elif not self.global_net and self.local_net:
            return None, None, [logits2, ca_act_regs2], crops

if __name__ == '__main__':
    from models.RACNN.loss import loc_loss
    net = PFAN_OD()
    # print(DEVICE)
    net.to(DEVICE)
    state_dict = torch.load(
        '/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_unet_saliency_detection/__tmp__/2020-12-21-21-34-50/checkpoints/checkpoint_150_1.pth',
        map_location=DEVICE
    )['state_dict']
    new_state_dict = {}
    # print(2)
    # for key in state_dict:
    #     new_state_dict[key.replace('unet.', '')] = state_dict[key]
    # net.unet.load_state_dict(new_state_dict)

    # state_dict = torch.load(
    #     '/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_detection/__tmp__/2021-01-11-14-35-41/checkpoints/checkpoint_5_1.pth',
    #     map_location=DEVICE
    # )['state_dict']
    # net.rcnn.rcnn.load_state_dict(
    #     state_dict
    # )
    # print(4)
    
    input = torch.randn((3, 3, 224, 224), requires_grad=True).to(DEVICE)
    final_outputs, outputs_1, outputs_2, crops = net(input)

    # print(outputs_1[0].shape)
    # print(outputs_2[0].shape)

    # print(ouput.shape)

    # loss = loc_loss([locs], [torch.randn(10, 6).to(DEVICE)])
    # loss.backward()

    # print(loss)
