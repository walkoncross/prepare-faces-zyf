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
import urllib
from skimage import io

#from matplotlib import pyplot as plt
from fx_warp_and_crop_face import get_reference_facial_points, warp_and_crop_face

#GT_RECT = [68, 68, 182, 182]
#GT_AREA = (GT_RECT[2] - GT_RECT[0] + 1) * (GT_RECT[3] - GT_RECT[1] + 1)
overlap_thresh = 0.3

only_align_missed = False
do_align = True

# crop settings, set the region of cropped faces
#output_square = True
#padding_factor = 0.25
#output_padding = (0, 0)
output_size = (96, 112)
#
# get the referenced 5 landmarks position in the crop settings
# reference_5pts = get_reference_facial_points(
#    output_size, padding_factor, output_padding, output_square)
reference_5pts = None

#landmark_fn = r'../lfw-mtcnn-fd-rlt/lfw-mtcnn-v2-matlab-fd-rlt-3imgs.json'
#landmark_fn = r'../lfw-mtcnn-fd-rlt/lfw_mtcnn_falied3_align_rlt.json'
#landmark_fn = r'../lfw-mtcnn-fd-rlt/lfw_mtcnn_fd_rlt_kirk_plus_failed3.json'
#img_root_dir = r'C:/zyf/dataset/lfw'

#landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw-mtcnn-v2-matlab-fd-rlt-3imgs.json'
#landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_falied3_align_rlt.json'
#landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_fd_rlt_kirk_plus_failed3.json'
#landmark_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_4nets_fd_rlt_add_missed.json'
#img_root_dir = r'/disk2/data/FACE/LFW/LFW'
#aligned_save_dir = img_root_dir + '-mtcnn-simaligned-96x112-new-4nets'
aligned_save_dir = './crop_data'

log_fn1 = 'align_succeeded_list-0.txt'
log_fn2 = 'align_failed_list-0.txt'
log_fn3 = 'faces_wrong_max_score_idx_list-0.txt'

file_root_path = '../../mtcnn-caffe-good/mtcnn_aligner/fd_json_add_missed/result_after_miss/'

log_align_params = 'align_params.txt'


def get_gt_overlap(faces,GT_RECT):
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
    
    GT_AREA = (GT_RECT[2] - GT_RECT[0] + 1) * (GT_RECT[3] - GT_RECT[1] + 1)   

    overlap = o_w * o_h

    overlap = overlap / (GT_AREA + area - overlap)

    return overlap


#def get_max_gt_overlap_face(faces, thresh=0.5):
#    overlap = get_gt_overlap(faces)
#    max_id = overlap.argmax()
#    if overlap[max_id] >= thresh:
#        return max_id
#    else:
#        return -1

#if only_align_missed:
#    print('Only process missed faces!!!')

if not osp.exists(file_root_path):
    print('ERROR: webface root dir not found!!!')

else:
    if not osp.exists(aligned_save_dir):
        print('mkdir for aligned faces, aligned root dir: ', aligned_save_dir)
        os.makedirs(aligned_save_dir)

    fp_log1 = open(osp.join(aligned_save_dir, log_fn1), 'w')
    fp_log2 = open(osp.join(aligned_save_dir, log_fn2), 'w')
    fp_log3 = open(osp.join(aligned_save_dir, log_fn3), 'w')

    failed_count1 = 0
    failed_count2 = 0

    fp = open('./summary_atflow_format.txt','r')
    all_lines = fp.readlines()
    count = 1
    for line in all_lines[:180000]:
        err_msg = ''
	print count
	count = count + 1
	data = json.loads(line)
		
	filepath = '%s/%s' % (data[u'url'].split('/')[-2],data[u'url'].split('/')[-1][:-4])
        filename = filepath + '.json'
        print osp.join(file_root_path,filename)
        result_file = open(osp.join(file_root_path,filename),'r')
        item = json.load(result_file)
        #item = result[0]
		
        if 'filename' not in item:
            err_msg = "'filename' not in item, break..."
            print(err_msg)
            fp_log2.write(err_msg + '\n')
            break

        #img_fn = osp.join(img_root_dir, item['filename'])
        print item['filename']
        splits = item['filename'].split('/')
        save_fn = osp.join(aligned_save_dir, splits[-2], splits[-1])
        save_fn_dir = osp.dirname(save_fn)

        overlap_thresh_0 = overlap_thresh

        print('===> Processing image: ' + filename)

        if 'faces' not in item:
            err_msg = "'faces' not in item"
            fp_log2.write(item['filename'] + ': ' + err_msg + '\n')
            continue
        elif 'face_count' not in item:
            err_msg = "'face_count' not in item"
            fp_log2.write(item['filename'] + ': ' + err_msg + '\n')
            continue

        if only_align_missed and 'used_gt' not in item:
            print('skipped because only_align_missed')
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
		
        GT_RECT = [item['gt'][0][0],item['gt'][0][1],item['gt'][2][0],item['gt'][2][1]]
        overlaps = get_gt_overlap(faces,GT_RECT)

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
                points = np.array(faces[max_overlap_idx]['pts'])
                facial5points = np.reshape(points, (2, -1))
                # print facial5points

                try:
                    #image = cv2.imread(img_fn, True)
                    imageBGR = io.imread(data[u'url'])
                    image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
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
