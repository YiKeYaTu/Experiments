import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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
                
                if xmax >= x[idx].shape[2]:
                    xmin -= xmax - (x[idx].shape[2] - 1)
                    xmax = x[idx].shape[2] - 1

            if height < 32:
                ymin -= math.ceil(padding_ho / 2)
                ymax += math.ceil(padding_ho / 2)

                if ymin < 0:
                    ymax -= ymin
                    ymin = 0
                
                if ymax >= x[idx].shape[1]:
                    ymin -= ymax - (x[idx].shape[1] - 1)
                    ymax = x[idx].shape[1] - 1

            crops.append({
                'box': (xmin, ymin, xmax, ymax),
                'image': x[idx][:, ymin: ymax, xmin: xmax]
            })

        logits2 = []
        features2 = []

        for crop in crops:
            xmin, ymin, xmax, ymax = crop['box']
            _logits2, _features2 = self.unet(crop['image'].unsqueeze(dim=0))
            logits2.append(
                self.sigmoid(_logits2)
            )
            features2.append(
                _features2
            )

        # logits2 = torch.stack(logits2)
        # features2 = torch.stack(features2)


        return logits2, crops


if __name__ == '__main__':
    from models.RACNN.loss import loc_loss
    net = RACNN()
    print(DEVICE)
    net.to(DEVICE)
    state_dict = torch.load(
        '/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_unet_saliency_detection/__tmp__/2020-12-21-21-34-50/checkpoints/checkpoint_150_1.pth',
        map_location=DEVICE
    )['state_dict']
    new_state_dict = {}
    print(2)
    for key in state_dict:
        new_state_dict[key.replace('unet.', '')] = state_dict[key]
    net.unet.load_state_dict(new_state_dict)

    state_dict = torch.load(
        '/home/ncrc-super/data/Liangchen/Experiments/tasks/pretrain_detection/__tmp__/2021-01-11-14-35-41/checkpoints/checkpoint_5_1.pth',
        map_location=DEVICE
    )['state_dict']
    net.rcnn.rcnn.load_state_dict(
        state_dict
    )
    print(4)
    
    input = torch.randn((10, 3, 224, 224), requires_grad=True).to(DEVICE)
    ouput, _, _ = net(input)

    # print(ouput.shape)

    # loss = loc_loss([locs], [torch.randn(10, 6).to(DEVICE)])
    # loss.backward()

    # print(loss)
