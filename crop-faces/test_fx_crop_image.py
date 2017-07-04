# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:32:00 2017

@author: zhaoy
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

from fx_crop_image import crop_square_image_into_ratio

img_fn = '../test_imgs/Jennifer_Aniston_0016.jpg'
print('Loading image {}'.format(img_fn))
image = cv2.imread(img_fn, True);

image_show = image[...,::-1]# swap BGR to RGB to show image by pyplot
plt.figure();
plt.imshow(image_show)


dst_img = crop_square_image_into_ratio(image, 7, 6)

dst_img_show = dst_img[...,::-1]# swap BGR to RGB to show image by pyplot
plt.figure()
plt.imshow(dst_img_show )

dst_img2 = crop_square_image_into_ratio(image, 6, 7)

dst_img2_show = dst_img2[...,::-1]# swap BGR to RGB to show image by pyplot
plt.figure()
plt.imshow(dst_img2_show )