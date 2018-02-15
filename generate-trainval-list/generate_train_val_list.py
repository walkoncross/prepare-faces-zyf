#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu May 04 03:31:51 2017

@author: zhaoy
"""

import os
import os.path as osp
import time
import random


root_dir = '../CASIA-maxpy-clean-aligned-zyf'
img_ext = '.jpg'
train_base_name = 'train_list'
val_base_name = 'val_list'

start_id = 0
val_ratio = 0.15

removed_ids = ['0166921', '1056413', '1193098']

train_save_name = train_base_name + '_ratio-' + str(1-val_ratio)
val_save_name = val_base_name + '_ratio-' + str(val_ratio)

root_dir = osp.abspath(root_dir)

path_walk = os.walk(root_dir)

train_file_list = []
val_file_list = []

i = start_id

train_cnt = 0
val_cnt = 0

train_ids_cnt = 0
val_ids_cnt = 0

for root,dirs,files in path_walk:
    removed_flag = 0
    for rid in removed_ids:
        if rid in root:
            removed_flag = 1
            break
    if removed_flag:
        continue

    if root!=root_dir and not root.startswith('.'):
        print('--->Processing dir: ' + str(root))
        tmp_list = []
        for f in files:
            if f.endswith(img_ext):
                tmp_list.append(f)

        if len(tmp_list)<1:
            continue

        ll = len(tmp_list)
        val_ll = int(ll*val_ratio)

        if val_ll<1:
            for f in tmp_list:
                train_file_list.append([osp.join(root, f), i])
                train_cnt += 1

            train_ids_cnt += 1
        else:
            all_idx = range(ll)
            random.shuffle(all_idx)
            val_idx = sorted(all_idx[0:val_ll])
            train_idx = sorted(all_idx[val_ll:])

            for k in val_idx:
                val_file_list.append([osp.join(root, tmp_list[k]), i])
                val_cnt += 1
            val_ids_cnt += 1

            for k in train_idx:
                train_file_list.append([osp.join(root, tmp_list[k]), i])
                train_cnt += 1
            train_ids_cnt += 1

        i+=1

#print('\n===>Final train_file_list: \n' + str(train_file_list))
#print('\n===>Final val_file_list: \n' + str(val_file_list))

ctime = time.strftime('%y%m%d-%H%M%S')
train_save_name = train_save_name + '_{}-ids_{}-objs_{}.txt'.format(train_ids_cnt, train_cnt, ctime)
val_save_name = val_save_name + '_{}-ids_{}-objs_{}.txt'.format(val_ids_cnt, val_cnt, ctime)

fp = open(train_save_name, 'w')
for it in train_file_list:
    fp.write('{} {}\n'.format(it[0], it[1]))
fp.close()

print('\n===>{} train ids with {} train files saved into : {}\n'.format(train_ids_cnt, train_cnt, train_save_name))

fp = open(val_save_name, 'w')
for it in val_file_list:
    fp.write('{} {}\n'.format(it[0], it[1]))
fp.close()

print('\n===>{} val ids with {} val files saved into : {}\n'.format(val_ids_cnt, val_cnt, val_save_name))
