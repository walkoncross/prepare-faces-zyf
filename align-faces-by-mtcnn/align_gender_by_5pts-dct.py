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
from skimage import io

#from matplotlib import pyplot as plt
from fx_warp_and_crop_face import get_reference_facial_points, warp_and_crop_face

#GT_RECT = [68, 68, 182, 182]
#GT_AREA = (GT_RECT[2] - GT_RECT[0] + 1) * (GT_RECT[3] - GT_RECT[1] + 1)
overlap_thresh = 0.3

only_align_missed = False
do_align = True

# crop settings, set the region of cropped faces
output_square = True
padding_factor = 0.25
output_padding = (0, 0)
output_size = (224, 224)

# get the referenced 5 landmarks position in the crop settings
reference_5pts = get_reference_facial_points(
    output_size, padding_factor, output_padding, output_square)

aligned_save_dir = './gender_crop_data'
file_root_path = '../../mtcnn-caffe-good/mtcnn_aligner/result_after_align/'


if not osp.exists(file_root_path):
    print('ERROR: webface root dir not found!!!')

else:
    if not osp.exists(aligned_save_dir):
        print('mkdir for aligned faces, aligned root dir: ', aligned_save_dir)
        os.makedirs(aligned_save_dir)
 
    count = 30049
    for root, dirs, files in os.walk(file_root_path):
        for file in files[30049:]:
            err_msg = ''
            print file
            json_fn = osp.join(file_root_path, file)
            print json_fn
	    f = open(json_fn,'r')
            count = count + 1
            print count
	    data=json.load(f)
            subdir=data["class"]
            filename = file[:-5]+'.jpg'
            #print filename
            save_fn = osp.join(aligned_save_dir,subdir,filename)
            #print save_fn
            save_fn_dir = osp.dirname(save_fn)
            print save_fn_dir
            if do_align and not osp.exists(save_fn_dir):
                os.makedirs(save_fn_dir)

            print('===> Processing image: ' + json_fn)

            points = np.array(data["detect"]["pts"])
            facial5points = np.reshape(points, (2, -1))

            imageBGR = io.imread(data[u'url'])
            image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB) 
            dst_img = warp_and_crop_face(image, facial5points, reference_5pts, output_size)
            cv2.imwrite(save_fn, dst_img)
