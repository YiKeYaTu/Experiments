import torchvision
import tasks.exdark_v1.detection.utils as utils
import torch
import tasks.exdark_v1.detection.transforms as T
import matplotlib.pyplot as plt
import os
import time

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tasks.exdark_v1.detection.engine import train_one_epoch, evaluate
from dataloaders.object_detection.ExDark import ExDark, OBJ_TYPES
from torch.utils.data import DataLoader
from constant import DEVICE, LEARNING_RATE, ITERATION_SIZE, WEIGHT_DECAY, TMP_ROOT
from torchvision.transforms.functional import to_pil_image
from dataloaders.object_detection.ExDark import OBJ_TYPES
from torchvision.models.detection.anchor_utils import AnchorGenerator


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

testloader = DataLoader(
    ExDark(
        train=False,
        transforms=get_transform(False)
    ),
    batch_size=10,
    shuffle=False,
    num_workers=8,
    collate_fn=utils.collate_fn
)

device = DEVICE

def get_model():
    # load a model pre-trained pre-trained on COCO
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        min_size=500, max_size=833,
        pretrained=True,
        progress=True,
        box_detections_per_img=58,
        image_mean=[0.17045, 0.1338, 0.2242],
        image_std=[0.17390, 0.1502, 0.13195],
        rpn_anchor_generator=rpn_anchor_generator
    )

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = len(OBJ_TYPES) + 1  # 12 class + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    return model

model = get_model()

def run():
    folder = os.path.join(TMP_ROOT, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    @torch.no_grad()
    def hook(images, targets, outputs, names):
        for idx in range(len(images)):
            image = images[idx]
            target = targets[idx]
            output = outputs[idx]

            fig = plt.imshow(to_pil_image(images[idx]))

            for box, label in zip(target['boxes'], target['labels']):
                target_box_plt = plt.Rectangle(
                    (box[0], box[1]),
                    width=box[2] - box[0],
                    height=box[3] - box[1],
                    fill=False,
                    edgecolor='g',
                    linewidth=2
                )
                fig.axes.add_patch(target_box_plt)
                plt.text(box[0], box[1], OBJ_TYPES[str(int(label))], fontdict={'size': 12, 'color': 'green'})

            for box, label in zip(output['boxes'], target['labels']):
                pred_box_plt = plt.Rectangle(
                    (box[0], box[1]),
                    width=box[2] - box[0],
                    height=box[3] - box[1],
                    fill=False,
                    edgecolor='r',
                    linewidth=2
                )
                fig.axes.add_patch(pred_box_plt)
                plt.text(box[0], box[1], OBJ_TYPES[str(int(label.data))], fontdict={'size': 12, 'color': 'red'})

            if not os.path.isdir(folder):
                os.makedirs(folder)

            plt.savefig(os.path.join(folder, 'test_%s.png' % names[idx]))
            plt.clf()
            break

    # evaluate on the test dataset
    evaluate(model, testloader, device=device, hook=hook)
