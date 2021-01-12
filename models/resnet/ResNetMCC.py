from models.resnet.ResNet import resnet18, resnet50, resnet101, resnet152
import torch.nn as nn

class ResNetMCC(nn.Module):

    def __init__(self, layer_count=18):
        super(ResNetMCC, self).__init__()

        if layer_count == 18:
            self.resnet = resnet18(pretrained=True)
        elif layer_count == 50:
            self.resnet = resnet50(pretrained=True)
        elif layer_count == 101:
            self.resnet = resnet101(pretrained=True)
        elif layer_count == 152:
            self.resnet = resnet152(pretrained=True)
        else:
            self.resnet = resnet18(pretrained=True)

        self.illu = nn.Conv2d(512 if resnet50 is False else 2048, 3, 1)
        self.upsample = nn.functional.interpolate

    def forward(self, x):
        _, _, h, w = x.size()

        x = self.resnet(x)
        x = self.illu(x)
        x = self.upsample(x, size=(h, w), mode='bilinear')

        return [ x ]
        
