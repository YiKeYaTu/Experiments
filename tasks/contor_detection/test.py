from dataloaders.ColorChecker_cv2 import ColorCheckerLoader
from models.RCF import RCF
from torch.utils.data import DataLoader
from constant import *
import torchvision
import torch
import os

fold1_train_loader = DataLoader(ColorCheckerLoader(
    fold_number=1, is_training=True), batch_size=1, shuffle=True, num_workers=8)

model = RCF()
model.to(DEVICE)
model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, '../Datasets/RCFcheckpoint_epoch12.pth'))['state_dict'])

for idx, (images, labels, names) in enumerate(fold1_train_loader):
    images = images.to(DEVICE)
    outputs = model(images)
    _, _, H, W = outputs[0].shape
    all_results = torch.zeros((len(outputs), 1, H, W))
    for j in range(len(outputs)):
        all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
    torchvision.utils.save_image(all_results, join(TMP_ROOT, "test_%s.jpg" % idx))
    torchvision.utils.save_image(images / 255, join(TMP_ROOT, "test_%s_.jpg" % idx))
    # cv2.imshow('sss', contor_maps[0])