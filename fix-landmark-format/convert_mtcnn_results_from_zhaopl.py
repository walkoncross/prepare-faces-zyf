# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 05:00:49 2017

@author: zhaoy
"""
import json

fn = './landmark_yrj_8imgs_wrong_correct.json'
fn_splits = fn.rsplit('.', 1)
fn_out = fn_splits[0] + '_new_format.' + fn_splits[1]

fp_in = open(fn, 'r')
fp_out = open(fn_out, 'w')

old_json_list = json.load(fp_in)

new_json_list = []

for it in old_json_list:
    total_faces_list = it['total_boxes']
    points_list = it['points']
    cnt = len(total_faces_list)

    new_faces_list = []
    if cnt>0:
        for i in range(cnt):
            new_faces_list.append(
                    {
                        'score': total_faces_list[i][4],
                        'rect': total_faces_list[i][:4],
                        'pts': points_list[i]
                    }
                    )
    new_item = {

            'filename': it['filename'],
            'id': it['id'],
            'face_count': cnt,
            'faces': new_faces_list,
            'message': "success"
            }
    new_json_list.append(new_item)

json.dump(new_json_list, fp_out, indent=4)

fp_in.close()
fp_out.close()