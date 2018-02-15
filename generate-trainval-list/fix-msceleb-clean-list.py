#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 2 2017

@author: zhaoy
"""

import os
import os.path as osp
import sys
import shutil


def get_correct_sub_dir(root_dir, sub_dir, fn):
    full_dir = osp.join(root_dir, sub_dir)
    if not osp.exists(full_dir):
        msg = 'Can not find dir ' + full_dir
        print(msg)
        # raise Exception(msg)
        return None

    dir_list = os.listdir(full_dir)
    sub_dir2 = None

    if len(dir_list) < 1:
        msg = 'No folder under dir ' + full_dir
        print(msg)
        # raise Exception(msg)
        return None
    else:
        cnt = 0
        for it in dir_list:
            if (osp.isdir(osp.join(full_dir, it)) and
                    osp.exists(osp.join(full_dir, it, fn))):
                sub_dir2 = it
                cnt += 1
                break
                # if cnt > 1:
                #     raise Exception('More than 1 folder under dir ' + full_dir)
        if cnt < 1:
            msg = 'No folder under dir ' + full_dir + ' contains: ' + fn
            print(msg)
            # raise Exception(msg)

    if sub_dir2:
        return osp.join(sub_dir, sub_dir2)
    else:
        return ''


def main(list_fn, root_dir=None):
    save_fn = osp.splitext(list_fn)[0] + '_fixed.txt'

    start_id = -1
    cnt = 0

    succ_cnt = 0
    fail_cnt = 0
    new_idx = 0

    fp = open(list_fn, 'r')
    fp_rlt = open(save_fn, 'w')

    print('\n===>save fixed list into file : {}\n'.format(save_fn))

    fixed_subdir = ''
    tmp_list = []

    for line in fp:
        spl = line.strip().split()
        fn_im_0 = spl[0]
        idx = int(spl[1])

        sub_dir, fn_im_1 = osp.split(fn_im_0)
        spl2 = fn_im_1.split('-')
        fn_im_2 = spl2[0] + '/face_' + spl2[-1]

        cnt += 1

        if idx != start_id:
            print(
                '\n---> processed {} image file name with {} old ids\n'.format(cnt - 1, start_id + 1))
            print('\tfailed to fix {} image file\n'.format(fail_cnt))
            print('\t succeded to fix {} image file with {} new_ids\n'.format(
                succ_cnt, new_idx + 1))

            if len(tmp_list) > 1:
                for it in tmp_list:
                    fp_rlt.write('%s %d\n' % (it, new_idx))

                fp_rlt.flush()
                new_idx += 1

            # reset list and get fixed_subdir
            tmp_list = []
            fixed_subdir = get_correct_sub_dir(root_dir, sub_dir, fn_im_2)
            start_id = idx

            if not fixed_subdir:
                fail_cnt += 1
                print('Cannot find fixed_subdir for: ' + fn_im_0)
            else:
                succ_cnt += 1
                fixed_fn = osp.join(fixed_subdir, fn_im_2)
                tmp_list.append(fixed_fn)
        else:
            # subdir does not exist
            if fixed_subdir is None:
                fail_cnt += 1
                print('Cannot find fixed_subdir for: ' + fn_im_0)
                continue

            file_found = False
            # use previous fixed_subdir if it exists
            if fixed_subdir:
                fixed_fn = osp.join(fixed_subdir, fn_im_2)
                full_fixed_fn = osp.join(root_dir, fixed_fn)

                if osp.exists(full_fixed_fn):
                    file_found = True
                    succ_cnt += 1
                    tmp_list.append(fixed_fn)

            # try to re-find fixed_subdir
            if not file_found:
                fixed_subdir = get_correct_sub_dir(root_dir, sub_dir, fn_im_2)

                if not fixed_subdir:
                    fail_cnt += 1
                    print('Cannot find fixed_subdir for: ' + fn_im_0)
                else:
                    succ_cnt += 1
                    fixed_fn = osp.join(fixed_subdir, fn_im_2)
                    tmp_list.append(fixed_fn)

    if len(tmp_list) > 1:
        for it in tmp_list:
            fp_rlt.write('%s %d\n' % (it, new_idx))

        fp_rlt.flush()

    fp.close()
    fp_rlt.close()

    print('\n===> processed {} image file name wit {} old ids\n'.formatcnt, start_id + 1)
    print('\t failed to fix {} image file\n'.format(fail_cnt))
    print('\t succeded to fix {} image file with {} new_ids\n'.format(
        succ_cnt, new_idx + 1))

    save_fn2 = osp.splitext(
        list_fn)[0] + '_fixed_%d_ids_%d_imgs.txt' % (new_idx + 1, succ_cnt)

    shutil.move(save_fn, save_fn2)
    print('\n===> rename {} int {}\n'.format(save_fn, save_fn2))


if __name__ == '__main__':
    root_dir = '/disk2/data/FACE/celeb-1m-mtcnn-aligned/msceleb_align/Faces-Aligned/'
    list_fn = ''

    if len(sys.argv) > 2:
        root_dir = sys.argv[2]

    if len(sys.argv) > 1:
        list_fn = sys.argv[1]

    main(list_fn, root_dir)
