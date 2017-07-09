# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 06:25:15 2017

@author: zhaoy
"""
import os
import sys
import os.path as osp

import numpy as np
from scipy.io import loadmat, savemat

ftr_path = r'C:/zyf/dnn_models/face_models/centerloss/lfw_eval_results/LFW_Feature.mat'
pairs_path = r'C:/zyf/dataset/lfw-txt/pairs.txt'

save_fn = './lfw-pairs-wyd.mat'

mat_data = loadmat(ftr_path)

tmp_list = mat_data['list']

ftr_name_list = []

for it in tmp_list:
    ftr_name_list.append(str(it[0][0]))

#print ftr_name_list

pos_pairs = []
neg_pairs = []

with open(pairs_path, 'r') as fp:
    fp.readline() # skip first line
    for line in fp:
        line = line.strip()
#        print line
        splits = line.split()
#        print len(splits)

        if len(splits) == 3:
            img1 = "%s_%04d.jpg" % (splits[0], int(splits[1]))
            img2 = "%s_%04d.jpg" % (splits[0], int(splits[2]))
#            print img1, img2
            pos_pairs.append((img1, img2))
        elif len(splits) == 4:
            img1 = "%s_%04d.jpg" % (splits[0], int(splits[1]))
            img2 = "%s_%04d.jpg" % (splits[2], int(splits[3]))
#            print img1, img2
            neg_pairs.append((img1, img2))

fp.close()

#print pos_pairs
#print neg_pairs


pos_pairs_idx_list = []
neg_pairs_idx_list = []

for it in pos_pairs:
    idx1 = ftr_name_list.index(it[0])
    idx2 = ftr_name_list.index(it[1])
    pos_pairs_idx_list.append([idx1, idx2])

for it in neg_pairs:
    idx1 = ftr_name_list.index(it[0])
    idx2 = ftr_name_list.index(it[1])
    neg_pairs_idx_list.append([idx1, idx2])

pos_pair = np.array(pos_pairs_idx_list)
neg_pair = np.array(neg_pairs_idx_list)

# convert into matlab index (starts from 1, not 0)
pos_pair = pos_pair + 1
neg_pair = neg_pair + 1

pos_pair = pos_pair.T
neg_pair = neg_pair.T

print pos_pair.shape
print neg_pair.shape

savemat(save_fn, {'pos_pair': pos_pair, 'neg_pair':neg_pair})