from dataloaders.ColorChecker import ColorCheckerLoader
from models.ColorConstancyNCRF import ColorConstancyNCRF
from torch.utils.data import DataLoader
from torchvision import transforms
from constant import *
from PIL import Image
import torch
import numpy as np

fold1_train_loader = DataLoader(ColorCheckerLoader(
    fold_number=1, is_training=True), batch_size=2, shuffle=True, num_workers=8)
fold1_test_loader = DataLoader(ColorCheckerLoader(
    fold_number=1, is_training=False), batch_size=10, shuffle=True, num_workers=8)

fold2_train_loader = DataLoader(ColorCheckerLoader(
    fold_number=2, is_training=True), batch_size=10, shuffle=True, num_workers=8)
fold2_test_loader = DataLoader(ColorCheckerLoader(
    fold_number=2, is_training=False), batch_size=10, shuffle=True, num_workers=8)

fold3_train_loader = DataLoader(ColorCheckerLoader(
    fold_number=3, is_training=True), batch_size=10, shuffle=True, num_workers=8)
fold3_test_loader = DataLoader(ColorCheckerLoader(
    fold_number=3, is_training=False), batch_size=10, shuffle=True, num_workers=8)

# cc = ColorCheckerLoader(fold_number=2, is_training=False)

# img, gt, name = cc.__getitem__(1)

# img = transforms.ToPILImage()(img)
# # print(img)
# img = np.array(img)
# img[:, :, :] = img[:, :, :] / gt / 3

# im = Image.fromarray(img)
# im.show()

model = ColorConstancyNCRF()
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(
    (parameter for parameter in model.parameters() if parameter.requires_grad is True),
    lr=LEARNING_RATE, momentum=0.9)

model.to(device=DEVICE)


def run(epoch):
    for idx, (images, labels, names) in enumerate(fold1_train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # img = images[0].cpu()
        # img = np.array(transforms.ToPILImage()(img))
        # img[:, :, :] = img[:, :, :] / np.array(labels[0].cpu())
        # # print(img.shape)
        # im = Image.fromarray(img)
        # im.show()

        illus = model(images)
        loss = criterion(illus, labels)
        loss.backward()

        print(loss.item())

        optimizer.zero_grad()
        optimizer.step()

        # exit()

    # for idx, (inputs, labels, names) in enumerate(fold1_train_loader):
    #     import torch
    #     outputs = model(torch.randn(1, 3, 100, 100))
    #     print(outputs)
    #     break
