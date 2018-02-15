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
import sys

#from matplotlib import pyplot as plt
import _init_paths
from fx_warp_and_crop_face import get_reference_facial_points, warp_and_crop_face


# crop settings, set the region of cropped faces
#default_square = True
#padding_factor = 0.25
#outer_padding = (0, 0)
output_size = (96, 112)
#
# get the referenced 5 landmarks position in the crop settings
# reference_5pts = get_reference_facial_points(
#    output_size, padding_factor, outer_padding, default_square)
reference_5pts = None

log_fn1 = 'align_succeeded_list.txt'
log_fn2 = 'align_failed_list.txt'
log_fn3 = 'faces_wrong_max_score_idx_list.txt'

log_align_params = 'align_params.txt'


def align_faces(landmark_fn, img_root_dir, aligned_save_dir, do_align=True):
    fp_in = open(landmark_fn, 'r')
    img_list = json.load(fp_in)
    fp_in.close()

    if not osp.exists(img_root_dir):
        print('ERROR: webface root dir not found!!!')

    else:
        if not osp.exists(aligned_save_dir):
            print('mkdir for aligned faces, aligned root dir: ', aligned_save_dir)
            os.makedirs(aligned_save_dir)

    #    fp_log_params = open(osp.join(aligned_save_dir, log_align_params), 'w')
    # params_template = '''
    ##    default_square = {}
    ##    padding_factor = {}
    ##    outer_padding = {}
    ##    output_size = {}
    # '''
    #    params_template = ('default_square = {}\n'
    #                       'padding_factor = {}\n'
    #                       'outer_padding = {}\n'
    #                       'output_size = {}\n')
    #
    #    fp_log_params.write(params_template.format(
    #            default_square, padding_factor,
    #            outer_padding, output_size)
    #    )
    #    fp_log_params.close()

        fp_log1 = open(osp.join(aligned_save_dir, log_fn1), 'w')
        fp_log2 = open(osp.join(aligned_save_dir, log_fn2), 'w')
        fp_log3 = open(osp.join(aligned_save_dir, log_fn3), 'w')

    #    imgSize = [112, 96];
    #    coord5points = [[30.2946, 65.5318, 48.0252, 33.5493, 62.7299],
    #                    [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]];
    #    pts_dst = np.float32(coord5points).transpose()

        failed_count1 = 0
        failed_count2 = 0

        for item in img_list:
            err_msg = ''
            if 'filename' not in item:
                err_msg = "'filename' not in item, break..."
                print(err_msg)
                fp_log2.write(err_msg + '\n')
                break

            img_fn = osp.join(img_root_dir, item['filename'])
            save_fn = osp.join(aligned_save_dir, osp.basename(item['filename']))
            save_fn_dir = osp.dirname(save_fn)

            print('===> Processing image: ' + img_fn)

            if 'faces' not in item:
                err_msg = "'faces' not in item"
                fp_log2.write(item['filename'] + ': ' + err_msg + '\n')
                continue
            elif 'face_count' not in item:
                err_msg = "'face_count' not in item"
                fp_log2.write(item['filename'] + ': ' + err_msg + '\n')
                continue

            if do_align and not osp.exists(save_fn_dir):
                os.makedirs(save_fn_dir)

            nfaces = item['face_count']

            if nfaces < 1:
                fp_log2.write(item['filename'] + ': ' +
                              "item['face_count'] < 1" + '\n')
                continue

            if nfaces != len(item['faces']):
                fp_log2.write(item['filename'] + ': ' +
                              "item['face_count'] != len(item['faces']" + '\n')
                continue

            faces = item['faces']
            scores = np.array([it['score'] for it in faces])
            max_score_idx = scores.argmax()

    #        max_score_idx = 0
    #
    #        if nfaces > 1:
    #            for idx in range(1, nfaces):
    #                if faces[idx]['score'] > faces[max_score_idx]['score']:
    #                    max_score_idx = idx

            max_overlap_idx = 0

            fp_log1.write(item['filename'] + ': ' + " max_overlap_idx="
                          + str(max_overlap_idx) + '\n')
            if do_align:
                points = np.array(faces[max_overlap_idx]['pts'])
                facial5points = np.reshape(points, (2, -1))
                # print facial5points

                try:
                    image = cv2.imread(img_fn, True)

                    dst_img = warp_and_crop_face(
                        image, facial5points, reference_5pts, output_size)
                    cv2.imwrite(save_fn, dst_img)
                except Exception as e:
                    failed_count1 += 1
                    fp_log2.write(item['filename'] + ': ' +
                                  "exception when loading image"
                                  " or aligning faces or saving results" + '\n')
                    fp_log2.write("\texception: {}".format(e) + '\n')
                    continue

                fp_log1.write(item['filename'] + ': ' +
                              " succeeded to align" + '\n')

        fp_log1.close()
        fp_log2.close()
        fp_log3.close()

    #        dst_img_show = dst_img[..., ::-1]
    #
    #        plt.figure()
    #        plt.imshow(dst_img_show)


if __name__ == '__main__':

    #landmark_fn = r'../lfw-mtcnn-fd-rlt/lfw-mtcnn-v2-matlab-fd-rlt-3imgs.json'
    landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_falied3_align_rlt.json'
    #landmark_fn = r'../lfw-mtcnn-fd-rlt/lfw_mtcnn_fd_rlt_kirk_plus_failed3.json'
    img_root_dir = r'C:/zyf/dataset/lfw'

    #landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw-mtcnn-v2-matlab-fd-rlt-3imgs.json'
    #landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_falied3_align_rlt.json'
    #landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_fd_rlt_kirk_plus_failed3.json'
    #landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_4nets_fd_rlt_add_missed.json'
    #img_root_dir = r'/disk2/data/FACE/LFW/LFW'
    aligned_save_dir = img_root_dir + '-mtcnn-simaligned-96x112'

    if len(sys.argv) > 1:
        landmark_fn = sys.argv[1]
    if len(sys.argv) > 2:
        img_root_dir = sys.argv[2]
    if len(sys.argv) > 3:
        aligned_save_dir = sys.argv[3]

    align_faces(landmark_fn, img_root_dir, aligned_save_dir)
