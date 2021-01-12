import torchvision

from torch.utils.data import Dataset
from constant import DATASETS_ROOT
from os import path, listdir
from PIL import Image

DATASET_ROOT_PATH = path.join(
    DATASETS_ROOT, 'Multi_Illu/Image')
GROUND_TRUTH_PATH = path.join(
    DATASETS_ROOT, 'Multi_Illu/Ground_Truth')

class ImageNet(Dataset):
    def __init__(
        self,
        train=True,
        transform=None,
        target_transform=None
    ):
        self.transform = transform
        self.train = train
        self.target_transform = target_transform

        self.image_paths = sorted(
            [path.join(DATASET_ROOT_PATH, p)
                for p in listdir(DATASET_ROOT_PATH)]
        )

        if self.train is True:
            self.image_paths = self.image_paths[0:-2000]
        else:
            self.image_paths = self.image_paths[-2000:]

        self.target_paths = [
            path.join(
                GROUND_TRUTH_PATH,
                p.split('/')[-1].replace('jpg', 'png').replace('color','gt')
            ) for p in self.image_paths
        ]
        
    def __getitem__(self, index):
        name = self.image_paths[index].split('/')[-1]
        image = Image.open(self.image_paths[index])
        target = Image.open(self.target_paths[index])

        assert image.size == target.size

        image = self.transform(image) \
            if self.transform is not None else image
        target = self.target_transform(target) \
            if self.target_transform is not None else target

        return image, target, name
    def __len__(self):
        return len(self.image_paths)
