from os import path, listdir
import cv2


def default_map(root, o):
    return path.join(root, o.split('/')[-1]).replace('.jpg', '.png')


def default_filter(img_name):
    if(img_name.lower().endswith((
        '.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'
    ))):
        return True
    
    return False


class ImagePairReader:
    def __init__(self, img_root=None, gt_root=None, map=default_map, filter=default_filter):
        self.img_root = img_root
        self.gt_root = gt_root
        self.map = map
        self.filter = filter

        self.img_list = sorted([
            path.join(self.img_root, p)
            for p in listdir(self.img_root) if self.filter(p) is True
        ])
        self.gt_list = [
            map(gt_root, p)
            for p in self.img_list
        ]

    def get_item(self, idx, reader=None):
        """
            Read input image by rgb mode as default.
            Read ground truth by gray mode as default.
        """
        img = reader(self.img_list[idx]) \
            if reader else cv2.cvtColor(cv2.imread(self.img_list[idx]), cv2.COLOR_BGR2RGB).astype('float32')
        gt = reader(self.gt_list[idx]) \
            if reader else cv2.imread(self.gt_list[idx], 0).astype('float32')

        return (
            self.img_list[idx].split('/')[-1],
            img,
            gt,
        )

    def get_len(self):
        return len(self.img_list)


if __name__ == '__main__':
    reader = ImagePairReader(
        '/home/ncrc-super/data/DataSets/saliency_deteciton/DUTS-TR/DUTS-TR-Image',
        '/home/ncrc-super/data/DataSets/saliency_deteciton/DUTS-TR/DUTS-TR-Mask'
    )
    print(reader.get_item(0))
