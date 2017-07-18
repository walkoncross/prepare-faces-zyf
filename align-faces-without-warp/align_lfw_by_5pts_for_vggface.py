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

from fx_image_roi import get_image_roi

GT_RECT = [68, 68, 182, 182]
GT_AREA = (GT_RECT[2] - GT_RECT[0] + 1) * (GT_RECT[3] - GT_RECT[1] + 1)
overlap_thresh = 0.3

do_align = True

# crop settings, set the region of cropped faces
output_square = True
padding_factor = 0.25
do_resize = True
output_size = (224, 224)

roi_scale = 1.0 + padding_factor * 2


#landmark_fn = r'../lfw-mtcnn-fd-rlt/lfw-mtcnn-v2-matlab-fd-rlt-3imgs.json'
landmark_fn = r'../lfw-mtcnn-fd-rlt/lfw_mtcnn_falied3_align_rlt.json'
#landmark_fn = r'../lfw-mtcnn-fd-rlt/lfw_mtcnn_fd_rlt_kirk_plus_failed3.json'
img_root_dir = r'C:/zyf/dataset/lfw'

#landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw-mtcnn-v2-matlab-fd-rlt-3imgs.json'
#landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_falied3_align_rlt.json'
#landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_fd_rlt_kirk_plus_failed3.json'
#img_root_dir = r'/disk2/data/FACE/LFW/LFW'

aligned_save_dir = img_root_dir + '-aligned-nowarp-224x224-new'

log_fn1 = 'align_succeeded_list.txt'
log_fn2 = 'align_failed_list.txt'
log_fn3 = 'faces_wrong_max_score_idx_list.txt'

log_align_params = 'align_params.txt'


def get_gt_overlap(faces):
    rects = [it['rect'] for it in faces]

    rects_arr = np.array(rects)
#    print 'rects_arr: {}'.format(rects_arr)
    area = (rects_arr[:, 2] - rects_arr[:, 0] + 1) * \
        (rects_arr[:, 3] - rects_arr[:, 1] + 1)
#    print 'area: {}'.format(area)

    o_x1 = np.maximum(GT_RECT[0], rects_arr[:, 0])
    o_x2 = np.minimum(GT_RECT[2], rects_arr[:, 2])
    o_y1 = np.maximum(GT_RECT[1], rects_arr[:, 1])
    o_y2 = np.minimum(GT_RECT[3], rects_arr[:, 3])

    o_w = np.maximum(0, o_x2 - o_x1 + 1)
    o_h = np.maximum(0, o_y2 - o_y1 + 1)

    overlap = o_w * o_h
#    print 'overlap area: {}'.format(overlap)

    overlap = overlap / (GT_AREA + area - overlap)
#    print 'overlap ratio: {}'.format(overlap)

    return overlap


def get_max_gt_overlap_face(faces, thresh=0.5):
    overlap = get_gt_overlap(faces)

    max_id = overlap.argmax()
#    print 'overlap[max_id]: %1.3f' % overlap[max_id]
    if overlap[max_id] >= thresh:
        return max_id
    else:
        return -1


fp_in = open(landmark_fn, 'r')
img_list = json.load(fp_in)
fp_in.close()

if not osp.exists(img_root_dir):
    print('ERROR: webface root dir not found!!!')

else:
    if not osp.exists(aligned_save_dir):
        print('mkdir for aligned faces, aligned root dir: ', aligned_save_dir)
        os.makedirs(aligned_save_dir)

    fp_log_params = open(osp.join(aligned_save_dir, log_align_params), 'w')
    params_template = ('output_square = {}\n'
                       'padding_factor = {}\n'
                       'do_resize = {}\n'
                       'output_size = {}\n')

    fp_log_params.write(params_template.format(
        output_square,
        padding_factor,
        do_resize,
        output_size)
    )
    fp_log_params.close()

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
        save_fn = osp.join(aligned_save_dir, item['filename'])
        save_fn_dir = osp.dirname(save_fn)

        overlap_thresh_0 = overlap_thresh

        # Tom_Brady_0002 is special cauz the face in the image is very small
        if 'Tom_Brady_0002' in img_fn:
            overlap_thresh_0 = 0.25

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

        overlaps = get_gt_overlap(faces)

        max_overlap_idx = overlaps.argmax()

        if max_score_idx != max_overlap_idx:
            fp_log3.write(item['filename'] + ': ' + '\n')
            fp_log3.write("--> max_score_idx   = {}\n".format(max_score_idx))
            fp_log3.write("--> max_overlap_idx = {}\n".format(max_overlap_idx))
            fp_log3.write("--> scores   = {}\n".format(scores))
            fp_log3.write("--> overlaps = {}\n".format(overlaps))

        if overlaps[max_overlap_idx] >= overlap_thresh_0:
            fp_log1.write(item['filename'] + ': ' + " max_overlap_idx="
                          + str(max_overlap_idx) + '\n')
            if do_align:
                rect = faces[max_overlap_idx]['rect']
                roi = [rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]]

                try:
                    image = cv2.imread(img_fn, True)

                    dst_img = get_image_roi(
                        image, roi, roi_scale, output_square)
                    if do_resize:
                        dst_img = cv2.resize(dst_img, output_size)
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
        else:
            failed_count2 += 1

            fp_log2.write(item['filename'] + ': ' +
                          "no faces have overlap>={} with groundtruth".format(
                              overlap_thresh_0) +
                          '\n')
            fp_log2.write("--> max_score_idx   = {}\n".format(max_score_idx))
            fp_log2.write("--> max_overlap_idx = {}\n".format(max_overlap_idx))
            fp_log2.write("--> scores   = {}\n".format(scores))
            fp_log2.write("--> overlaps = {}\n".format(overlaps))

    fp_log2.write("\n==>Faied images: {}\n".format(failed_count1
                                                   + failed_count2))
    fp_log2.write("\t{} failed because of exception\n".format(failed_count1))
    fp_log2.write("\t{} failed because of max_overlap<thresh\n".format(
        failed_count2))

    fp_log1.close()
    fp_log2.close()
    fp_log3.close()

#        dst_img_show = dst_img[..., ::-1]
#
#        plt.figure()
#        plt.imshow(dst_img_show)
