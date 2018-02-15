#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 05:25:59 2017

@author: zhaoy
"""

def merge_train_lists(fn1, fn2, fn_o):
    fp_o = open(fn_o, 'w')
    fp1 = open(fn1, 'r')
    max_id = 0
    for line in fp1:
        fp_o.write(line)
        t_id = int(line.strip().split()[-1])
        if max_id < t_id:
            max_id = t_id

    fp1.close()

    max_id += 1

    fp2 = open(fn2, 'r')
    for line in fp2:
        spl = line.strip().split()
        t_fn = spl[0]
        t_id = int(spl[-1]) + max_id
        fp_o.write('{} {:d}\n'.format(t_fn, t_id))
    fp2.close()
    fp_o.close()


if __name__=='__main__':
    fn1 = '/disk2/zhaoyafei/centerface-resnet-prototxt-noval/train_list_noval_10572-ids_450833-objs_170503-213839.txt'
    fn2 = '/disk2/zhaoyafei/sphereface-train-asian/face_asian_train_list_noval_10245-ids_540735-objs_170818-225846.txt'
    fn_o = './train_list_webface_asian_20817-ids_991568-objs_171025.txt'

    merge_train_lists(fn1, fn2, fn_o)