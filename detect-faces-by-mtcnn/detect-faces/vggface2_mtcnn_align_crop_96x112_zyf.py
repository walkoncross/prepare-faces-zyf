#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 06:21:16 2017

@author: zhaoy
"""

import numpy as np
import cv2
import os
import os.path as osp
import sys

# reload(sys)
# sys.setdefaultencoding("utf-8")

# from matplotlib import pyplot as plt
from mtcnn_aligner import MtcnnAligner

from fx_warp_and_crop_face import get_reference_facial_points, warp_and_crop_face


output_size = (96, 112)
reference_5pts = None


def print_usage():
    usage = 'python %s <nsplits> <split_id>  <img-list-file> <img-root-dir> [<MTCNN-model-dir>] [<save-dir>]' % osp.basename(
        __file__)
    print('USAGE: ' + usage)


def main(nsplits, split_id, list_file, img_root_dir, mtcnn_model_dir, save_dir=None):
    if not save_dir:
        save_dir = './aligned_root_dir'

    if not osp.exists(save_dir):
        print('mkdir for aligned root dir: ', save_dir)
        os.makedirs(save_dir)

    save_aligned_dir = osp.join(save_dir, 'aligned_imgs')
    if not osp.exists(save_aligned_dir):
        print('mkdir for aligned/cropped face imgs: ', save_dir)
        os.makedirs(save_aligned_dir)

    save_rects_dir = osp.join(save_dir, 'face_rects')
    if not osp.exists(save_rects_dir):
        print('mkdir for face rects/landmarks: ', save_rects_dir)
        os.makedirs(save_rects_dir)

    aligner = MtcnnAligner(mtcnn_model_dir, False)

    fp = open(list_file, 'r')
    all_lines = fp.readlines()
    fp.close()

    total_line_cnt = len(all_lines)
    print('--->%d imgs in total' % total_line_cnt)

    if nsplits < 2:
        if split_id > 0:
            print('===> Will only process first %d imgs' % split_id)
            start_line = 0
            end_line = split_id
        else:
            print('===> Will process all of the images')
            start_line = 0
            end_line = total_line_cnt
    else:
        assert(split_id < nsplits)
        lines_per_split = float(total_line_cnt) / nsplits
        start_line = int(lines_per_split * split_id)
        end_line = int(lines_per_split * (split_id + 1))
        if end_line + 1 >= total_line_cnt:
            end_line = total_line_cnt

        print('===> Will only process imgs in the range [%d, %d)]' % (
            start_line, end_line))

    count = start_line

    for line in all_lines[start_line:end_line]:
        line = line.strip()
        print count

        count = count + 1
        img_fn = osp.join(img_root_dir, line)

        print('===> Processing img: ' + img_fn)
        img = cv2.imread(img_fn)
        ht = img.shape[0]
        wd = img.shape[1]

        print 'image.shape:', img.shape
        # GT_RECT = [0,0,img.shape[0],img.shape[1]]
        GT_RECT = [int(wd * 0.25), int(ht * 0.25),
                   int(wd * 0.75), int(ht * 0.72)]

        # print 'face rect: ', gt
        boxes, points = aligner.align_face(img, [GT_RECT])

        box = boxes[0]
        pts = points[0]

        spl = osp.split(line)
        sub_dir = spl[0]
        base_name = spl[1]

        save_img_subdir = osp.join(save_aligned_dir, sub_dir)
        if not osp.exists(save_img_subdir):
            os.mkdir(save_img_subdir)

        save_rect_subdir = osp.join(save_rects_dir, sub_dir)
        if not osp.exists(save_rect_subdir):
            os.mkdir(save_rect_subdir)
        # print pts

        save_img_fn = osp.join(save_img_subdir, base_name)

        facial5points = np.reshape(pts, (2, -1))
        dst_img = warp_and_crop_face(
            img, facial5points, reference_5pts, output_size)
        cv2.imwrite(save_img_fn, dst_img)

        save_rect_fn = osp.join(
            save_rect_subdir, osp.splitext(base_name)[0] + '.txt')
        fp_rect = open(save_rect_fn, 'w')
        for it in box:
            fp_rect.write('%5.2f\t' % it)
        fp_rect.write('\n')

        for i in range(5):
            fp_rect.write('%5.2f\t%5.2f\n' %
                          (facial5points[0][i], facial5points[1][i]))
        fp_rect.close()


if __name__ == "__main__":
    print_usage()
    mtcnn_model_dir = '../model'

    list_fn = '/disk2/data/FACE/vggface2/meta/train_list.txt'
    img_root_dir = '/disk2/data/FACE/vggface2/train/'
    split_id = 10
    nsplits = 0
    save_dir = './vggface2_train_aligned'

    print(sys.argv)

    if len(sys.argv) > 1:
        nsplits = int(sys.argv[1])

    if len(sys.argv) > 2:
        split_id = int(sys.argv[2])

    if len(sys.argv) > 3:
        list_fn = sys.argv[3]

    if len(sys.argv) > 4:
        img_root_dir = sys.argv[4]

    if len(sys.argv) > 5:
        mtcnn_model_dir = int(sys.argv[5])

    if len(sys.argv) > 6:
        save_dir = sys.argv[6]

    main(nsplits, split_id, list_fn, img_root_dir, mtcnn_model_dir, save_dir)
