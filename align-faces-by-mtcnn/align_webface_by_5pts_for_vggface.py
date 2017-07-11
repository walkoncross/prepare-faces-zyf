#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 01 05:18:38 2017

@author: zhaoy
"""

import numpy as np
import cv2
import json
import os
import os.path as osp

#from matplotlib import pyplot as plt
from fx_warp_and_crop_face import get_reference_facial_points, warp_and_crop_face

output_square = True
padding_factor = 0.25
output_padding = (0, 0)
output_size = (224, 224)

reference_5pts = get_reference_facial_points(
    output_size, padding_factor, output_padding, output_square)


landmark_fn = r'./landmark_yrj_8imgs.json'
webface_src_dir = r'C:/zyf/dataset/webface/CASIA-maxpy-clean'
#landmark_fn = r'../../webface-mtcnn-fd-rlt/landmark_yrj_8imgs.json'
#webface_src_dir = r'/disk2/data/FACE/webface/CASIA-maxpy-clean'
aligned_save_dir = webface_src_dir + '-simaligned-vggface'

log_fn1 = 'align_succeeded_list.txt'
log_fn2 = 'align_failed_list.txt'

fp_in = open(landmark_fn, 'r')
img_list = json.load(fp_in)
fp_in.close()

if not osp.exists(webface_src_dir):
    print('ERROR: webface root dir not found!!!')

else:
    if not osp.exists(aligned_save_dir):
        print('mkdir for aligned faces, aligned root dir: ', aligned_save_dir)
        os.makedirs(aligned_save_dir)

    fp_log1 = open(osp.join(aligned_save_dir, log_fn1), 'w')
    fp_log2 = open(osp.join(aligned_save_dir, log_fn2), 'w')

#    imgSize = [112, 96];
#    coord5points = [[30.2946, 65.5318, 48.0252, 33.5493, 62.7299],
#                    [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]];
#    pts_dst = np.float32(coord5points).transpose()


    for item in img_list:
        err_msg = ''
        if 'filename' not in item:
            err_msg = "'filename' not in item, break..."
            print(err_msg)
            fp_log2.write(err_msg + '\n')
            break

        img_fn = osp.join(webface_src_dir, item['filename'])
        save_fn = osp.join(aligned_save_dir, item['filename'])
        save_fn_dir = osp.dirname(save_fn)

        print('===> Processing image: ' + img_fn)

        if 'total_boxes' not in item:
            err_msg = "'total_boxes' not in item"
            fp_log2.write(item['filename'] + ': ' + err_msg +'\n')
            continue
        elif 'points' not in item:
            err_msg = "'points' not in item"
            fp_log2.write(item['filename'] + ': ' + err_msg +'\n')
            continue

        if not osp.exists(save_fn_dir):
            os.makedirs(save_fn_dir)

        nfaces = len(item['total_boxes'])

        if nfaces < 1:
            fp_log2.write(item['filename'] + ': ' + "nfaces < 1"+'\n')
            continue

        if nfaces != len(item['points']):
            fp_log2.write(item['filename'] + ': ' + "nfaces != len(item['points']"+'\n')
            continue

        max_idx = 0

        if nfaces>1:
            for idx in range(2, nfaces):
                if item['total_boxes'][idx][4] > item['total_boxes'][max_idx][4]:
                    max_idx = idx

        points = np.array(item['points'][max_idx])
        facial5points = np.reshape(points, (2, -1))

        try:
            image = cv2.imread(img_fn, True);

            dst_img = warp_and_crop_face(image, facial5points, reference_5pts, output_size)
            cv2.imwrite(save_fn, dst_img)
        except Exception as e:
            fp_log2.write(item['filename'] + ': ' +
                          "exception when loading image or aligning faces or saving results"+'\n')
            fp_log2.write("\texception: {}".format(e) +'\n')
            continue

        fp_log1.write(item['filename'] + ': ' + " succeeded to align"+'\n')

    fp_log1.close()
    fp_log2.close()

#        dst_img_show = dst_img[..., ::-1]
#
#        plt.figure()
#        plt.imshow(dst_img_show)