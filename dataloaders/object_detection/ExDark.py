
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from os import path
from torch.utils.data import Dataset

if __name__ == '__main__':
    import sys
    import os
    abspath = os.path.dirname(__file__)
    sys.path.append(os.path.join(abspath, '../../'))

from constant import DATASETS_ROOT

IMG_PATH = path.join(
    DATASETS_ROOT, 'ExDark/ExDark')
DETECTION_GT_PATH = path.join(
    DATASETS_ROOT, 'ExDark/ExDark_DetectionAnno')

OBJ_TYPES = {
    '1': 'Bicycle',
    '2': 'Boat',
    '3': 'Bottle',
    '4': 'Bus',
    '5': 'Car',
    '6': 'Cat',
    '7': 'Chair',
    '8': 'Cup',
    '9': 'Dog',
    '10': 'Motorbike',
    '11': 'People',
    '12': 'Table',
}

OBJ_LABELS = {
    'Bicycle': 1,
    'Boat': 2,
    'Bottle': 3,
    'Bus': 4,
    'Car': 5,
    'Cat': 6,
    'Chair': 7,
    'Cup': 8,
    'Dog': 9,
    'Motorbike': 10,
    'People': 11,
    'Table': 12,
}

SUFFIXS = ('PNG', 'png', 'JPG', 'jpg', 'JPEG', 'jpeg')
IGNORE_LIST = ('2015_05894.jpg')


class ExDark(Dataset):
    def __init__(
        self,
        train=True,
        augment=True,
        transforms=None,
    ):
        self.train = train
        self.transforms = transforms

        self.imageclasslist = self._read_imageclasslist()
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )

    def _read_imageclasslist(self):
        fd = open(path.join(DATASETS_ROOT, 'ExDark/imageclasslist.txt'), 'r')
        lst = fd.readlines()[1:]
        result = []

        for line in lst:
            line = line.replace('\n', '').split(' ')
            f = False

            if (self.train and int(line[4]) != 3) or (not self.train and int(line[4]) == 3):
                f = True

            if f:
                temp = {
                    'image_name': line[0],
                    'image_path': path.join(IMG_PATH, OBJ_TYPES[line[1]], line[0]),
                    'detection_label_path': path.join(DETECTION_GT_PATH, OBJ_TYPES[line[1]], line[0] + '.txt'),
                    'object_class_label': line[1],
                    'object_segmentation_label': None,
                    'lighting_type': line[2],
                    'scene': line[3],
                }

                if line[0] in IGNORE_LIST:
                    continue

                if not path.isfile(temp['image_path']):
                    print('Skip image: %s, bacause image does not exsits.' % line[0])
                    continue
                if not path.isfile(temp['detection_label_path']):
                    for suffix in SUFFIXS:
                        next_path = path.join(DETECTION_GT_PATH, OBJ_TYPES[line[1]], line[0].split('.')[0] + '.' + suffix + '.txt')
                        print(next_path)
                        if path.isfile(next_path):
                            temp['detection_label_path'] = next_path
                            break

                if not path.isfile(temp['detection_label_path']):
                    print('Skip image: %s' % line[0])
                    continue

                result.append(temp)

        return result

    def _read_target(self, idx):
        cur = self.imageclasslist[idx]
        tar_path = cur['detection_label_path']

        fd = open(tar_path, 'r')

        lst = fd.readlines()[1:]
        num_objs = len(lst)

        boxes = []
        labels = []
        masks = []

        for item in lst:
            obj_type, l, t, w, h, *_ = item.replace('\n', '').split(' ')
            boxes.append([float(l), float(t), float(
                l) + float(w), float(t) + float(h)])
            labels.append(float(OBJ_LABELS[obj_type]))

            if cur['object_segmentation_label']:
                pass

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.float32)

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([idx])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        if cur['object_segmentation_label']:
            target['masks'] = masks

        return target

    def __getitem__(self, idx):
        cur = self.imageclasslist[idx]

        image = Image.open(cur['image_path']).convert("RGB")
        target = self._read_target(idx)
        name = cur['image_name']

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, name

    def __len__(self):
        return len(self.imageclasslist)
# mean [0.17045, 0.1338, 0.2242]
# std  [0.17390, 0.1502, 0.13195]
def compute_mean_std(train=True):
    dataset = ExDark(train=train)
    mean = list([])
    std = list([])

    for idx in range(len(dataset)):
        img, _, _ = dataset.__getitem__(idx)
        img = torch.from_numpy(np.array(img, dtype=np.float)) / 255
        img = img.reshape(-1, 3)

        mean.append(torch.mean(img, dim=0))
        std.append(torch.std(img ,dim=0))

    mean = torch.stack(mean)
    std = torch.stack(std)

    return torch.mean(mean, dim=0), torch.mean(std, dim=0)


if __name__ == '__main__':
    print('train dataset: ', compute_mean_std(True))
    print('test dataset: ', compute_mean_std(False))
    # dataset = ExDark()
    # max_box = 0

    # mean = []

    # for i in range(len(dataset)):
    #     img, target, name = dataset.__getitem__(i)
    #     img = torch.from_numpy(np.array(img, dtype=np.float)) / 255
    #     img = img.reshape(-1, 3)
    #     print(img.shape)
    #     print(torch.mean(img, dim=0))
    #     break

    #     number_box = len(target['boxes'])

    #     if number_box > max_box:
    #         max_box = number_box
    #     break

    # print('Max box: ', max_box)
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # empty = np.array(img)
    # for box in target['boxes']:
    #     xmin, ymin, xmax, ymax = box
    #     empty[int(ymin):int(ymax), int(xmin):int(xmax)] = 255
    # plt.imshow(empty)
    # plt.show()

    # print(img)
    # print(target)
