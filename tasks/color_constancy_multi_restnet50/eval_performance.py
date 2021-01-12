
from dataloaders.multi_color_constancy.ImageNet import ImageNet
from models.resnet.ResNetMCC import ResNetMCC
from torch.utils.data import DataLoader
from constant import DEVICE, TMP_ROOT
from utils.StatisticalValue import StatisticalValue
from loss_functions.multi_angular_loss import multi_angular_loss
from torchvision import transforms
import torch
import torchvision
import os
import time
from thop import profile

dataset = ImageNet(
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor()
    ]),
    target_transform=transforms.Compose([
        transforms.ToTensor()
    ]),
)
testloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=8
)

model = ResNetMCC(resnet50=True)
model.to(device=DEVICE)

macs, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(DEVICE), ))
print("Model's macs is %f, params is %f" % (macs, params))