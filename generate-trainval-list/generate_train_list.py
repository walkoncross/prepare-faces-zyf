#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 03 14:39:29 2017

@author: zhaoy
"""
import os
import os.path as osp
import time

root_dir = './'
img_ext = '.prototxt'
save_name = 'train_list.txt'
start_id = 0

if osp.exists(save_name):
    splits = save_name.rsplit('.', 1)
    ctime = time.strftime('%y%m%d-%H%M%S')

    save_name = splits[0] + '_' + ctime + splits[1]

root_dir = osp.abspath(root_dir)

path_walk = os.walk(root_dir)

file_list = []

i = start_id

for root,dirs,files in path_walk:
    if root!=root_dir and not root.startswith('.'):
        print('--->Processing dir: ' + str(root))
        tmp_list = []
        for f in files:
            if f.endswith(img_ext):
                tmp_list.append(f)

        if len(tmp_list)<1:
            continue

        for f in tmp_list:
            file_list.append([osp.join(root, f), i])

        i+=1

print('\n===>Final file_list: \n' + str(file_list))


fp = open(save_name, 'w')
for it in file_list:
    fp.write('{} {}\n'.format(it[0], it[1]))

fp.close()
