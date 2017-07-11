# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:32:00 2017

@author: zhaoy
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

from fx_warp_and_crop_face import warp_and_crop_face, get_reference_facial_points

img_fn = '../test_imgs/Jennifer_Aniston_0016.jpg'
# imgSize = [96, 112]; # cropped dst image size

# facial points in cropped dst image
#    coord5points = [[30.2946, 65.5318, 48.0252, 33.5493, 62.7299],
#                    [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]];

# facial points in src image
# facial5points = [[105.8306, 147.9323, 121.3533, 106.1169, 144.3622],
#                 [109.8005, 112.5533, 139.1172, 155.6359, 156.3451]];
facial5points = [[105.8306,  109.8005],
                 [147.9323,  112.5533],
                 [121.3533,  139.1172],
                 [106.1169,  155.6359],
                 [144.3622,  156.3451]
                 ]

print('Loading image {}'.format(img_fn))
image = cv2.imread(img_fn, True)

# for pt in src_pts[0:3]:
#    cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), 1, 8, 0)

# swap BGR to RGB to show image by pyplot
image_show = image[..., ::-1]

plt.title('src image')
plt.figure()
plt.imshow(image_show)


def crop_test(image,
              facial5points,
              reference_points=None,
              output_size=(96, 112),
              align_type='similarity'):

    #dst_img = transform_and_crop_face(image, facial5points, coord5points, imgSize)

    dst_img = warp_and_crop_face(image,
                                 facial5points,
                                 reference_points,
                                 output_size,
                                 align_type)

    print 'warped image shape: ', dst_img.shape

    # swap BGR to RGB to show image by pyplot
    dst_img_show = dst_img[..., ::-1]

    plt.figure()
    plt.title(align_type + ' transform ' + str(output_size))
    plt.imshow(dst_img_show)


print '===>test default crop setting with similarity transform'
crop_test(image, facial5points)

print '===>test default crop setting with cv2 affine transform'
crop_test(image, facial5points, align_type='affine')

print '===>test default crop setting with default affine transform'
crop_test(image, facial5points, align_type='affine')

print '===>test default square crop setting with similarity transform'
# crop settings, set the region of cropped faces
output_square = True
inner_padding_factor = 0.25
output_padding = (0, 0)
output_size = (224, 224)

# get the reference 5 landmarks position in the crop settings
reference_5pts = get_reference_facial_points(
    output_size, inner_padding_factor, output_padding, output_square)
print '--->reference_5pts:\n', reference_5pts

print '===>test default square crop setting with similarity transform'
crop_test(image, facial5points, reference_5pts, output_size)

print '===>test default square crop setting with similarity transform'
crop_test(image, facial5points, reference_5pts, output_size,
          align_type='cv2_affine')

print '===>test default crop setting with default affine transform'
crop_test(image, facial5points, reference_5pts, output_size,
          align_type='affine')
