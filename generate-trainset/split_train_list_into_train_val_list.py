#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Nov 16 2017

@author: zhaoy
"""

import os
import os.path as osp
import time
import random


original_train_list = '/disk2/zhaoyafei/sphereface-train-merge_webface_asian/train_list_webface_asian_20817-ids_991568-objs_171025.txt'

base_name = 'webface_plus_asian'
train_base_name = base_name + '_train_list'
val_base_name = base_name + '_val_list'

start_id = -1
val_ratio = 0.05

# removed_ids = ['0166921', '1056413', '1193098']

tmp_train_save_fn = train_base_name + '-tmp.txt'
tmp_val_save_fn = val_base_name + '-tmp.txt'

ctime = time.strftime('%y%m%d-%H%M%S')

train_save_name = train_base_name + '_ratio-' + str(1 - val_ratio)
val_save_name = val_base_name + '_ratio-' + str(val_ratio)

train_cnt = 0
val_cnt = 0

train_ids_cnt = 0
val_ids_cnt = 0

fp = open(original_train_list, 'r')

fp_train = open(tmp_train_save_fn, 'w')
fp_val = open(tmp_val_save_fn, 'w')

print('\n===>save train_list into tmp file : {}\n'.format(tmp_train_save_fn))
print('\n===>save val_list into tmp file : {}\n'.format(tmp_val_save_fn))

tmp_list = []

for line in fp:
    spl = line.strip().split()
    fn_im = spl[0]
    idx = int(spl[1])

    if start_id < 0:
        start_id = idx
        tmp_list.append(fn_im)
        continue

    if idx == start_id:
        tmp_list.append(fn_im)
        continue
    else:
        ll = len(tmp_list)
        val_ll = int(ll * val_ratio)

        if val_ll < 1:
            for f in tmp_list:
                fp_train.write('%s %d\n' % (f, idx))
                train_cnt += 1

            train_ids_cnt += 1
        else:
            all_idx = range(ll)
            random.shuffle(all_idx)
            val_idx = sorted(all_idx[0:val_ll])
            train_idx = sorted(all_idx[val_ll:])

            for k in val_idx:
                fp_val.write('%s %d\n' % (tmp_list[k], start_id))
                val_cnt += 1
            val_ids_cnt += 1

            for k in train_idx:
                fp_train.write('%s %d\n' % (tmp_list[k], start_id))
                train_cnt += 1
            train_ids_cnt += 1

        tmp_list = []
        tmp_list.append(fn_im)
        start_id = idx

        fp_train.flush()
        fp_val.flush()

if tmp_list:
    all_idx = range(ll)
    random.shuffle(all_idx)
    val_idx = sorted(all_idx[0:val_ll])
    train_idx = sorted(all_idx[val_ll:])

    for k in val_idx:
        fp_val.write('%s %d\n' % (tmp_list[k], start_id))
        val_cnt += 1
    val_ids_cnt += 1

    for k in train_idx:
        fp_train.write('%s %d\n' % (tmp_list[k], start_id))
        train_cnt += 1
    train_ids_cnt += 1

fp_train.close()
fp_val.close()

train_save_name = train_save_name + '_{}-ids_{}-objs_{}.txt'.format(
    train_ids_cnt, train_cnt, ctime)

val_save_name = val_save_name + '_{}-ids_{}-objs_{}.txt'.format(
    val_ids_cnt, val_cnt, ctime)

print('\n===>rename tmp file {} into {}\n'.format(
    tmp_train_save_fn, train_save_name))
print('\n===>rename tmp file {} into {}\n'.format(
    tmp_val_save_fn, val_save_name))

os.rename(tmp_train_save_fn, train_save_name)
os.rename(tmp_val_save_fn, val_save_name)

print('\n===>{} train ids with {} train files saved into : {}\n'.format(
    train_ids_cnt, train_cnt, train_save_name))

print('\n===>{} val ids with {} val files saved into : {}\n'.format(
    val_ids_cnt, val_cnt, val_save_name))
