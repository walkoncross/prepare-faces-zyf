# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 05:00:49 2017

@author: zhaoy
"""
import json
import numpy as np
import re

fn = './landmark_yrj_8imgs_wrong_correct_new_format.json'
# fn = '../../webface-mtcnn-fd-rlt/landmark_correct.json'

fn_splits = fn.rsplit('.', 1)
fn_out = fn_splits[0] + '_2.' + fn_splits[1]

fp_in = open(fn, 'r')
fp_out = open(fn_out, 'w')

old_json_list = json.load(fp_in)

new_json_list = []

for it in old_json_list:
    total_faces_list = []
    scores_list = []
    points_list = []
    cnt = it['face_count']

    new_faces_list = []
    if cnt > 0:
        for face in it['faces']:
            total_faces_list.append(face['rect'])

            scores_list.append(face['score'])
            points_list.append(face['pts'])

        total_faces_array = np.array(total_faces_list).reshape(cnt, -1)
        scores_array = np.array(scores_list).reshape(cnt, -1)
        points_array = np.array(points_list).reshape(cnt, -1)
        total_faces_array = np.hstack((total_faces_array, scores_array))

        total_faces_list=total_faces_array.tolist()
        points_list=points_array.tolist()

    new_item={
            'filename': it['filename'],
            'id': it['id'],
            'face_count': cnt,
            'boxes': total_faces_list,
            'points': points_list,
            'message': "success"
            }
    new_json_list.append(new_item)

#json.dump(new_json_list, fp_out, indent=1)
s = json.dumps(new_json_list, indent=2)
s = re.sub('\s*{\s*"(.)": (\d+),\s*"(.)": (\d+)\s*}(,?)\s*', r'{"\1":\2,"\3":\4}\5', s)

fp_out.write(s)

fp_in.close()
fp_out.close()
