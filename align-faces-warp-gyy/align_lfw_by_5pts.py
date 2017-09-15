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
from face_db_align import AlignImage

# crop settings, set the region of cropped faces
output_square = True
padding_factor = 0.25
output_padding = (0, 0)
output_size = (128, 128)

img_root_dir = 'C:\zyf\dataset\lfw'
landmark_fn = r'../lfw-mtcnn-fd-rlt/lfw_mtcnn_falied3_align_rlt.json'
#landmark_fn = r'./mtcnn_fd_rlt_test_imgs.json'

#landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw-mtcnn-v2-matlab-fd-rlt-3imgs.json'
#landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_falied3_align_rlt.json'
#landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_fd_rlt_kirk_plus_failed3.json'
#landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_4nets_fd_rlt_add_missed.json'
#img_root_dir = r'/disk2/data/FACE/LFW/LFW'

#save_dir_prefix = img_root_dir
save_dir_prefix = './faces'

aligned_save_dir = save_dir_prefix + '-mtcnn-gyy-aligned-128x128'
aligned_save_dir2 = save_dir_prefix + '-mtcnn-gyy-aligned-96x112'

log_fn1 = 'align_succeeded_list.txt'
log_fn2 = 'align_failed_list.txt'

log_align_params = 'align_params.txt'

fp_in = open(landmark_fn, 'r')
img_list = json.load(fp_in)
fp_in.close()


face_aligner = AlignImage()

if not osp.exists(aligned_save_dir):
    print('mkdir for aligned faces, aligned root dir: ', aligned_save_dir)
    os.makedirs(aligned_save_dir)

    fp_log_params = open(osp.join(aligned_save_dir, log_align_params), 'w')
    params_template = '''
    output_square = {}
    padding_factor = {}
    output_padding = {}
    output_size = {}
    '''

    fp_log_params.write(params_template.format(
            output_square, padding_factor,
            output_padding, output_size)
    )
    fp_log_params.close()

if not osp.exists(aligned_save_dir2):
    print('mkdir for aligned faces, aligned root dir2: ', aligned_save_dir2)
    os.makedirs(aligned_save_dir2)

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

    img_fn = osp.join(img_root_dir, item['filename'])

    #save_fn = osp.join(aligned_save_dir, osp.basename(item['filename']))
    save_fn = osp.join(aligned_save_dir,item['filename'])
    save_fn_dir = osp.dirname(save_fn)

    #save_fn2 = osp.join(aligned_save_dir2, osp.basename(item['filename']))
    save_fn2 = osp.join(aligned_save_dir2, item['filename'])
    save_fn_dir2 = osp.dirname(save_fn2)

    print('===> Processing image: ' + img_fn)

    if 'faces' not in item:
        err_msg = "'faces' not in item"
        fp_log2.write(item['filename'] + ': ' + err_msg + '\n')
        continue
    elif 'face_count' not in item:
        err_msg = "'face_count' not in item"
        fp_log2.write(item['filename'] + ': ' + err_msg + '\n')
        continue

    if not osp.exists(save_fn_dir):
        os.makedirs(save_fn_dir)

    if not osp.exists(save_fn_dir2):
        os.makedirs(save_fn_dir2)

    nfaces = item['face_count']

    if nfaces < 1:
        fp_log2.write(item['filename'] + ': ' + "item['face_count'] < 1" + '\n')
        continue

    if nfaces != len(item['faces']):
        fp_log2.write(item['filename'] + ': ' +
                        "item['face_count'] != len(item['faces']" + '\n')
        continue

    faces = item['faces']
    max_idx = 0

    if nfaces > 1:
        for idx in range(1, nfaces):
            if faces[idx]['score'] > faces[max_idx]['score']:
                max_idx = idx

    points = np.array(faces[max_idx]['pts'])
    #facial5points = np.reshape(points, (2, -1))

    try:
        image = cv2.imread(img_fn, True)

        dst_img = face_aligner.align(image, [points])[0]
        cv2.imwrite(save_fn, dst_img)

        dst_img2 = cv2.resize(dst_img, (96, 112))
        cv2.imwrite(save_fn2, dst_img2)

    except:
        fp_log2.write(item['filename'] + ': ' +
                        "exception when loading image or aligning faces or saving results" + '\n')
        continue

    fp_log1.write(item['filename'] + ': ' + " succeeded to align" + '\n')

fp_log1.close()
fp_log2.close()

#        dst_img_show = dst_img[..., ::-1]
#
#        plt.figure()
#        plt.imshow(dst_img_show)
