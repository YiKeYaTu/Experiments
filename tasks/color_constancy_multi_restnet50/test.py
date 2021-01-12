
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

# macs, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(DEVICE), ))
# print("Model's macs is %f, params is %f" % (macs, params))

def run():
    statistical_angular_errors = StatisticalValue()
    sub_dir = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print('Test start.')

    with torch.no_grad():
        for idx, (images, labels, names) in enumerate(testloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            predictions = model(images)

            angular_error = multi_angular_loss(predictions[-1], labels)
            statistical_angular_errors.update(angular_error.item(), names, sort=True)

            view_data = torch.zeros((4, *images.shape[1:]))
            view_data[0, :, :, :] = images.squeeze()
            view_data[1, :, :, :] = images.squeeze() / predictions[-1].squeeze()
            view_data[2, :, :, :] = predictions[-1].squeeze()
            view_data[3, :, :, :] = labels.squeeze()

            if not os.path.isdir(os.path.join(TMP_ROOT, 'test', sub_dir)):
                os.makedirs(os.path.join(TMP_ROOT, 'test', sub_dir))

            torchvision.utils.save_image(
                view_data,
                os.path.join(TMP_ROOT, 'test/%s/%s' % (sub_dir, names[0]))
            )

            print(
                'Angular Error: mean: {errors.avg}, mid: {errors.mid}, worst: {errors.max[0]}, best: {errors.min[0]}'.format(
                    errors=statistical_angular_errors))

    print('Test end.')
