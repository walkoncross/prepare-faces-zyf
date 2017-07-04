# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:58:35 2017

@author: zhaoy
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from fx_transform_and_crop_face import transform_and_crop_face

import caffe

img_fn = '../test_imgs/Jennifer_Aniston_0016.jpg'
fn_splits = img_fn.rsplit('.', 1)
save_fn = fn_splits[0] + '_warped.' + fn_splits[1]
#
#
#clear;clc;
#% addpath('path_to_matCaffe/matlab');
#% caffe.reset_all();
#
#% load face model and creat network
#% caffe.set_device(0);
#% caffe.set_mode_gpu();
#% model = 'path_to_deploy/face_deploy.prototxt';
#% weights = 'path_to_model/face_model.caffemodel';
#% net = caffe.Net(model, weights, 'test');
#
#% load face image, and align to 112 X 96
#imgSize = [112, 96];
#coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299; ...
#                51.6963, 51.5014, 71.7366, 92.3655, 92.2041];
#
#image = imread('./Jennifer_Aniston_0016.jpg');
#facial5points = [105.8306, 147.9323, 121.3533, 106.1169, 144.3622; ...
#                 109.8005, 112.5533, 139.1172, 155.6359, 156.3451];

#% load face image, and align to 112 X 96
imgSize = [112, 96];
coord5points = [[30.2946, 65.5318, 48.0252, 33.5493, 62.7299],
                [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]];

face_bbox = [
                    [84, 58],
                    [174, 58],
                    [174, 185],
                    [84, 185]
                ]

pts_dst = np.float32(coord5points).transpose()

facial5points = [[105.8306, 147.9323, 121.3533, 106.1169, 144.3622],
                 [109.8005, 112.5533, 139.1172, 155.6359, 156.3451]];

pts_src = np.float32(facial5points).transpose()

image = cv2.imread(img_fn, True);

#for pt in pts_src[0:3]:
#    cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), 1, 8, 0)
cv2.line(image, tuple(face_bbox[0]), tuple(face_bbox[1]), (255, 255, 0), 1, 8, 0)
cv2.line(image, tuple(face_bbox[1]), tuple(face_bbox[2]), (255, 255, 0), 1, 8, 0)
cv2.line(image, tuple(face_bbox[2]), tuple(face_bbox[3]), (255, 255, 0), 1, 8, 0)
cv2.line(image, tuple(face_bbox[0]), tuple(face_bbox[3]), (255, 255, 0), 1, 8, 0)

image_show = image[...,::-1]

plt.figure();
plt.imshow(image_show)

#
#Tfm =  cp2tform(facial5points', coord5points', 'similarity');
#cropImg = imtransform(image, Tfm, 'XData', [1 imgSize(2)],...
#                                  'YData', [1 imgSize(1)], 'Size', imgSize);
#
#pts_dst_1 = np.float32(pts_dst[0:3])
#pts_src_1 = np.float32(pts_src[0:3])
#tfm = cv2.getAffineTransform(pts_dst_1, pts_src_1)
#tfm = cv2.getAffineTransform(pts_dst[0:3], pts_src[0:3])
tfm = cv2.getAffineTransform(pts_src[0:3], pts_dst[0:3])
print tfm

#figure;
#imshow(image);
#
#figure;
#imshow(cropImg);
#
#% transform image, obtaining the original face and the horizontally flipped one
#% if size(cropImg, 3) < 3
#%    cropImg(:,:,2) = cropImg(:,:,1);
#%    cropImg(:,:,3) = cropImg(:,:,1);
#% end
#% cropImg = single(cropImg);
#% cropImg = (cropImg - 127.5)/128;
#% cropImg = permute(cropImg, [2,1,3]);
#% cropImg = cropImg(:,:,[3,2,1]);
#%
#% cropImg_(:,:,1) = flipud(cropImg(:,:,1));
#% cropImg_(:,:,2) = flipud(cropImg(:,:,2));
#% cropImg_(:,:,3) = flipud(cropImg(:,:,3));
dst_img = cv2.warpAffine(image, tfm, (imgSize[1], imgSize[0]))


for pt in pts_dst[0:3]:
    cv2.circle(dst_img, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), 1, 8, 0)


dst_img_show = dst_img[...,::-1]

plt.figure()
plt.imshow(dst_img_show)

dst_img2 = transform_and_crop_face(image, facial5points)
cv2.imwrite(save_fn, dst_img2)

dst_img_show2 = dst_img2[..., ::-1]

plt.figure()
plt.imshow(dst_img_show2)


#caffe_img = caffe.io.load_image(img_fn)
#plt.figure()
#plt.imshow(caffe_img)

#% extract deep feature
#% res = net.forward({cropImg});
#% res_ = net.forward({cropImg_});
#% deepfeature = [res{1}; res_{1}];
#%
#% caffe.reset_all();