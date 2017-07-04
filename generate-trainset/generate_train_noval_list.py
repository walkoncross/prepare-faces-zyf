#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 03 14:39:29 2017

@author: zhaoy
"""
import os
import os.path as osp
import time

root_dir = '../CASIA-maxpy-clean-aligned-zyf'
img_ext = '.jpg'
train_base_name = 'train_list_noval'

start_id = 0

removed_ids = ['0166921', '1056413', '1193098']

train_save_name = train_base_name

root_dir = osp.abspath(root_dir)

path_walk = os.walk(root_dir)

train_file_list = []

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

        for f in tmp_list:
            train_file_list.append([osp.join(root, f), i])
            train_cnt += 1

        train_ids_cnt += 1

        i+=1

#print('\n===>Final train_file_list: \n' + str(train_file_list))
#print('\n===>Final val_file_list: \n' + str(val_file_list))

ctime = time.strftime('%y%m%d-%H%M%S')
train_save_name = train_save_name + '_{}-ids_{}-objs_{}.txt'.format(train_ids_cnt, train_cnt, ctime)

fp = open(train_save_name, 'w')
for it in train_file_list:
    fp.write('{} {}\n'.format(it[0], it[1]))
fp.close()

print('\n===>{} train ids with {} train files saved into : {}\n'.format(train_ids_cnt, train_cnt, train_save_name))
