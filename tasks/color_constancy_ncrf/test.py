from dataloaders.ColorChecker_cv2 import ColorCheckerLoader
from models.ColorConstancyNCRF import ColorConstancyNCRF
from torch.utils.data import DataLoader
from constant import *
from PIL import Image
import torch
import numpy as np
import torchvision

fold1_test_loader = DataLoader(ColorCheckerLoader(
    fold_number=1, is_training=True), batch_size=1, shuffle=True, num_workers=8)

model = ColorConstancyNCRF()
model.to(device=DEVICE)

for idx, (images, labels, names) in enumerate(fold1_test_loader):
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    illus = model(images)
    images, illus = images.cpu().detach().numpy().transpose(
        0, 2, 3, 1), illus.cpu().detach().numpy()
    illus = illus[:, np.newaxis, np.newaxis, :]

    images = images[:, :, :, :] / illus

    result = Image.fromarray((images[0]).astype(np.uint8))
    result.save(join(TMP_ROOT, "corrected_%s.png") % names[0])

    print("Running test [%d/%d]" % (idx + 1, len(fold1_test_loader)))
