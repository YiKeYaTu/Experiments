import torchvision
import tasks.pretrain_detection.utils as utils
import torch

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tasks.pretrain_detection.engine import train_one_epoch, evaluate
from dataloaders.saliency_detection.DUTS_detection import DUTS
from torch.utils.data import DataLoader
from constant import DEVICE, LEARNING_RATE, ITERATION_SIZE, WEIGHT_DECAY, TMP_ROOT

trainloader = DataLoader(
    DUTS(
        train=True
    ),
    batch_size=10,
    shuffle=False,
    num_workers=8,
    collate_fn=utils.collate_fn
)
testloader = DataLoader(
    DUTS(
        train=False
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8,
    collate_fn=utils.collate_fn
)

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True,
    progress=True,
    box_detections_per_img=100,
)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = DEVICE
model.to(device)

criterion = None

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

def run(epoch):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, trainloader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    if epoch % 5 == 0:
        try:
            evaluate(model, testloader, device=device)
        except:
            print('Test error.')
