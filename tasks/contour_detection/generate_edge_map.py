from dataloaders.ColorChecker import ColorCheckerLoader
from models.RCF import RCF
from torch.utils.data import DataLoader
from constant import *
import torchvision
import torch
import os

test_loader = DataLoader(
    ColorCheckerLoader(
        fold_number=None,
        is_training=False,
        input_transform=torchvision.transforms.functional.to_tensor
    ),
    batch_size=1,
    shuffle=False,
    num_workers=1
)

model = RCF()
model.to(DEVICE)
model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'pths/RCFcheckpoint_epoch12.pth'))['state_dict'])

for idx, (images, labels, names) in enumerate(test_loader):
    with torch.no_grad():
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        labels = labels.unsqueeze(2).unsqueeze(3)

        outputs = model(images / labels)
        _, _, H, W = outputs[0].shape

        images, labels, outputs = images.cpu(), labels.cpu(), outputs.cpu()

        for idx, (output) in enumerate(outputs[3:]):
            output = output.squeeze()
            torchvision.utils.save_image(
                output,
                join(
                    DATASETS_ROOT,
                    'boundaries',
                    'boundary%s_%s' % (idx, names[0])
                )
            )
            print("%s has been saved." % names[0])

        torch.cuda.empty_cache()

    # all_results = torch.zeros((len(outputs), 1, H, W))
    # for j in range(len(outputs)):
    #     all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
    # torchvision.utils.save_image(all_results, join(TMP_ROOT, "test_%s.jpg" % idx))
    # torchvision.utils.save_image(images / 255, join(TMP_ROOT, "test_%s_.jpg" % idx))
    # cv2.imshow('sss', contor_maps[0])