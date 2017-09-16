# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 06:21:16 2017

@author: zhaoy
"""

import numpy as np
import cv2
import json
import os
import os.path as osp
import sys
from skimage import io

#from matplotlib import pyplot as plt
from mtcnn_aligner import MtcnnAligner

def print_usage():
    usage = 'python %s <img-det-json-file> <lfw-root-dir> [<MTCNN-model-dir>] [<save-dir>]' % osp.basename(__file__)
    print('USAGE: ' + usage)

def main(mtcnn_model_dir, save_dir=None):
    if save_dir is None:
        save_dir = './result_after_align'

    if not osp.exists(save_dir):
        print('mkdir for aligned faces, aligned root dir: ', save_dir)
        os.makedirs(save_dir)

    aligner = MtcnnAligner(mtcnn_model_dir, False)

    
    fp = open('./dataset-gender-all.json','r')
    all_lines = fp.readlines()
    count = 25499
    for line in all_lines[25499:]:
        err_msg = ''
 
        count = count + 1
        print count
        data = json.loads(line)

        image_url = data["url"]
        image_bbox = data["label"]["detect"]["general_d"]["bbox"]        
        #"label":{"detect":{"general_d":{"bbox":[{"class":"male","pts":
        print image_url
        print image_bbox
        #print len(image_bbox)
  
        for idx in range(len(image_bbox)):
            if image_bbox[idx]==[]:
                continue
            else:
                img = io.imread(image_url)
                if len(img.shape)!=3:
                    continue
                else:
                    file_idx = "%06d" % count
                    print file_idx
                    filename = str(file_idx)+'_'+str(idx)+'.json'
                    fp_rlt = open(osp.join(save_dir,filename),'w')
                    item = {}
                    print('===> Processing image: ' + filename)
                    per_bbox=image_bbox[idx]["pts"]
                    print "bbox:",per_bbox
                    GT_RECT = [per_bbox[0][0],per_bbox[0][1],per_bbox[2][0],per_bbox[2][1]]
                    print "GT:",GT_RECT
                
                    boxes, points = aligner.align_face(img, [GT_RECT])

                    box = boxes[0]
                    pts = points[0]
                    item['url']=image_url
                    item['class']=image_bbox[idx]["class"]
                    tmp = {'rect': GT_RECT,'pts': pts}
                    item['detect']=tmp
                    print "item:",item
                    json.dump(item, fp_rlt, indent=4)
                    fp_rlt.close()


if __name__ == "__main__":
    print_usage()
    mtcnn_model_dir = '../model'
    save_dir = None

    print(sys.argv)

    if len(sys.argv) > 1:
        mtcnn_model_dir = sys.argv[1]

    if len(sys.argv) > 2:
        save_dir = sys.argv[2]

    main(mtcnn_model_dir, save_dir)
