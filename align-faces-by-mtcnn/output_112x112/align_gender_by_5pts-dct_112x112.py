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
# from skimage import io

#from matplotlib import pyplot as plt
import _init_paths
from fx_warp_and_crop_face import get_reference_facial_points, warp_and_crop_face

#GT_RECT = [68, 68, 182, 182]
#GT_AREA = (GT_RECT[2] - GT_RECT[0] + 1) * (GT_RECT[3] - GT_RECT[1] + 1)
overlap_thresh = 0.3

only_align_missed = False
do_align = True

# crop settings, set the region of cropped faces
default_square = True
padding_factor = 0
outer_padding = (0, 0)
output_size = (112, 112)

# get the referenced 5 landmarks position in the crop settings
reference_5pts = get_reference_facial_points(
    output_size, padding_factor, outer_padding, default_square)

aligned_save_dir = '/disk2/data/FACE/gender-lablex/gender_mtcnn_simaligned_112x112'
# json_root_path = '/disk2/data/FACE/gender-lablex/face-rects-json-mtcnn'
json_root_path = '/disk2/data/FACE/gender-lablex/gender_original_labelx_20170808/label'
img_root_dir = '/disk2/data/FACE/gender-lablex/gender_original_labelx_20170808/img'

if not osp.exists(json_root_path):
    print('ERROR: webface root dir not found!!!')

else:
    if not osp.exists(aligned_save_dir):
        print('mkdir for aligned faces, aligned root dir: ', aligned_save_dir)
        os.makedirs(aligned_save_dir)

    start_cnt = 0
    count = start_cnt
    for root, dirs, files in os.walk(json_root_path):
        for file in files[start_cnt:]:
            err_msg = ''
            json_fn = osp.join(json_root_path, file)
            print '===> Load json file: ', json_fn
            f = open(json_fn, 'r')
            count = count + 1
            print count
            data = json.load(f)
            f.close()

            img_fn = data['imgname']
            base_name = osp.splitext(img_fn)[0]
            image = cv2.imread(osp.join(img_root_dir), img_fn)

            for i, det in enumerate(data['detect']):
                print('---> Processing face #%d', i)
                subdir = det["class"]
                filename = base_name + '_%d.jpg' % i
                # print filename
                save_fn = osp.join(aligned_save_dir, subdir, filename)
                # print save_fn
                save_fn_dir = osp.dirname(save_fn)
                # print save_fn_dir
                if do_align and not osp.exists(save_fn_dir):
                    os.makedirs(save_fn_dir)


                points = np.array(det["pts"])
                facial5points = np.reshape(points, (2, -1)).T

                # imageBGR = io.imread(data[u'url'])
                # image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
                dst_img = warp_and_crop_face(
                    image, facial5points, reference_5pts, output_size)
                cv2.imwrite(save_fn, dst_img)
