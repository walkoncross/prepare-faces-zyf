# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 06:25:15 2017

@author: zhaoy
"""
import os
import sys
import os.path as osp

import json

import numpy as np
from scipy.io import loadmat, savemat

fd_rlt_path = r'C:/zyf/github/FaceVerification/dataset/lfw-mtcnn-v2-fd-rlt.mat'
list_path = r'C:/zyf/github/lfw-evaluation-zyf/lfw_data/lfw_list_mtcnn.txt'
output_json = r'./lfw-mtcnn-v2-matlab-fd-rlt.json'

mat_data = loadmat(fd_rlt_path)

tmp_list = mat_data['image_list']
bbox_list = mat_data['bbox_list']
points_list = mat_data['points_list']

mat_name_list = []

for it in tmp_list:
    img_fn = str(it[0][0])
    splits = osp.split(img_fn)
    splits2 = osp.split(splits[0])

    img_fn = '%s/%s' % (splits2[-1], splits[-1])

    mat_name_list.append(img_fn)

#print ftr_name_list

results = []

with open(list_path, 'r') as fp:
    for line in fp:
        line = line.strip()
#        print line
        splits = line.split()
#        print len(splits)

        if len(splits) == 2:
            img_fn = splits[0]
            idx = mat_name_list.index(img_fn)

            bbox = bbox_list[idx, :]
            points = points_list[idx, :]

            rlt = {}
            rlt["filename"] = img_fn
            tmp = {'rect': bbox[0:4].tolist(),
                   'score': bbox[4].tolist(),
                   'pts': points.tolist()
                   }
            rlt['faces'] = [tmp]
            rlt['id'] = splits[1]
            rlt['face_count'] = 1
            rlt['message'] = 'success'

            results.append(rlt)

fp.close()

fp_out = open(output_json, 'w')
json.dump(results, fp_out, indent=4)
fp_out.close()
