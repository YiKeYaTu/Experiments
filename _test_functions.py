from constant import *
# from PIL import Image, ImageFilter

# im = Image.open('tasks/contor_detection/__tmp__/test_dev/test_1.jpg')
# im = im.filter(ImageFilter.GaussianBlur(radius=20))
# im.show()

from dataloaders.ColorChecker import _test

_test()
# _gen_aug_images()

# from utils.functions.edge import get_edge_map
# from PIL import Image, ImageFilter

# get_edge_map(Image.open('/Volumes/LiangChen/Experiments/tasks/contour_detection/__tmp__/test_dev/gt_0.jpg'))

# import cv2
# import numpy as np
import os
# from matplotlib import pyplot as plt

# img = cv2.imread(os.path.join(DATASETS_ROOT, 'ColorChecker_Recommended/552_IMG_0884.tiff'), 0)

# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

# plt.show()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv2.imread(os.path.join(DATASETS_ROOT, 'ColorChecker_Recommended/552_IMG_0884.tiff'))

# blur = cv2.blur(img,(400,400))

# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()