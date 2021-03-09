import torchvision
import tasks.exdark_v1.detection.utils as utils
import torch
import tasks.exdark_v1.detection.transforms as T
import os
from PIL import Image, ImageDraw, ImageFont
from dataloaders.object_detection.ExDark import ExDark, OBJ_TYPES
from torch.utils.data import DataLoader
from constant import TMP_ROOT

trainloader = DataLoader(
    ExDark(
        train=True,
    ),
    batch_size=8,
    shuffle=False,
    num_workers=8,
    collate_fn=utils.collate_fn
)
testloader = DataLoader(
    ExDark(
        train=False,
    ),
    batch_size=8,
    shuffle=False,
    num_workers=8,
    collate_fn=utils.collate_fn
)


def hook(images, targets, names):
    for idx in range(len(images)):
        image = images[idx]
        target = targets[idx]
        name = names[idx]

        draw = ImageDraw.Draw(image)

        for iidx, (box) in enumerate(target['boxes']):

            draw.text(
                (box[0] + 4, box[1] + 4),
                OBJ_TYPES[str(int(target['labels'][iidx]))],
            )
            draw.rectangle(
                (box[0], box[1], box[2], box[3]),
                outline='green',
                width=2
            )

        print(os.path.join(TMP_ROOT, '%s' % (name)))
        image.save(os.path.join(TMP_ROOT, '%s' % (name)))


for idx, (images, targets, names) in enumerate(testloader):
    hook(images, targets, names)
