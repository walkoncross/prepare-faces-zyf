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

reload(sys)
sys.setdefaultencoding( "utf-8" )

#from matplotlib import pyplot as plt
from mtcnn_aligner import MtcnnAligner

from fx_warp_and_crop_face import get_reference_facial_points, warp_and_crop_face

def print_usage():
    usage = 'python %s <img-det-json-file> <lfw-root-dir> [<MTCNN-model-dir>] [<save-dir>]' % osp.basename(__file__)
    print('USAGE: ' + usage)

overlap_thresh = 0.3

output_size = (96, 112)
reference_5pts = None
aligned_save_dir = './megaface_align'

def get_gt_overlap(faces,GT_RECT):
    rects = [it['rect'] for it in faces]

    rects_arr = np.array(rects)
    print rects_arr

    area = (rects_arr[:, 2] - rects_arr[:, 0] + 1) * \
        (rects_arr[:, 3] - rects_arr[:, 1] + 1)
    o_x1 = np.maximum(GT_RECT[0], rects_arr[:, 0])
    o_x2 = np.minimum(GT_RECT[2], rects_arr[:, 2])
    o_y1 = np.maximum(GT_RECT[1], rects_arr[:, 1])
    o_y2 = np.minimum(GT_RECT[3], rects_arr[:, 3])

    o_w = np.maximum(0, o_x2 - o_x1 + 1)
    o_h = np.maximum(0, o_y2 - o_y1 + 1)

    GT_AREA = (GT_RECT[2] - GT_RECT[0] + 1) * (GT_RECT[3] - GT_RECT[1] + 1)

    overlap = o_w * o_h
    overlap = overlap / (GT_AREA + area - overlap)

    return overlap


def get_max_gt_overlap_face(faces, thresh=0.5):
    overlap = get_gt_overlap(faces)

    max_id = overlap.argmax()

    if overlap[max_id] >= thresh:
        return max_id
    else:
        return -1

def main(face_json_file, mtcnn_model_dir, save_dir=None):
    if not osp.exists(save_dir):
        print('mkdir for aligned faces, aligned root dir: ', save_dir)
        os.makedirs(save_dir)

    if not osp.exists(aligned_save_dir):
        print('mkdir for aligned faces, aligned root dir: ', save_dir)
        os.makedirs(aligned_save_dir)

    aligner = MtcnnAligner(mtcnn_model_dir, False)

    fp = open('./identity_json_list_new.json','r')
    all_lines = fp.readlines()
    print len(all_lines)
    count = 4375705 
    for line in all_lines[count:4400000]:
        err_msg = ''
        print count
        #print line
        count = count + 1
        data = json.loads(line)

        #print data
        #origin_image_url = data[u'url']

        image_url = data[u'url']
        #splits = origin_image_url.split('/')
        #image_url = splits[0] + '//' + osp.join(splits[2],splits[3],splits[4],splits[6],splits[7],splits[8])
        #print '*******',image_url

        gt = data[u'label'][u'detect'][u'general_d'][u'bbox'][0][u'pts']
        print image_url        

        overlap_thresh_0 = overlap_thresh
       
        save_path = osp.join(data[u'url'].split('/')[-3],data[u'url'].split('/')[-2])
        #print save_path
        isExists1 = osp.exists(osp.join(save_dir,save_path))
        if not isExists1:
            os.makedirs(osp.join(save_dir,save_path))

        save_img_fn = osp.join(data[u'url'].split('/')[-3],data[u'url'].split('/')[-2])
        save_fn =aligned_save_dir + '/' + save_img_fn + '/' + data[u'url'].split('/')[-1]
        #print save_fn
        isExists2 = osp.exists(osp.join(aligned_save_dir,save_img_fn))
        if not isExists2:
            os.makedirs(osp.join(aligned_save_dir,save_img_fn))

        filename = data[u'url'].split('/')[-1][:-4] + '.json'
        print filename

        fp_rlt = open(osp.join(save_dir,save_path,filename), 'w')
        item = {}
        item[u'url'] = image_url

        try:
        #print('===> Processing image: ' + save_path + '/' + filename)

            imageBGR = io.imread(image_url)
            image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
 
            print image.shape
            #GT_RECT = [0,0,image.shape[0],image.shape[1]]
            #GT_RECT = [item['gt'][0][0],item['gt'][0][1],item['gt'][2][0],item['gt'][2][1]]
            GT_RECT = [gt[0][0],gt[0][1],gt[2][0],gt[2][1]]

        #print 'gt',gt
        #print 'GT_RECT',GT_RECT
            boxes, points = aligner.align_face(image, [GT_RECT])

            box = boxes[0]
            pts = points[0]

        #print pts

            facial5points = np.reshape(points, (2, -1))
            dst_img = warp_and_crop_face(image, facial5points, reference_5pts, output_size)
            cv2.imwrite(save_fn, dst_img)

        #print '########'            
            print('===> Processing image: ' + save_path + '/' + filename)

            tmp = {'rect': box[0:4],
                   'score': box[4],
                   'gt': gt,
                   'pts': pts
                  }
            item['faces'] = tmp
            item['id'] = data[u'url'].split('/')[-2]
            item['shape'] = image.shape
        except:
            continue
        json.dump(item, fp_rlt, indent=4)
        fp_rlt.close()

if __name__ == "__main__":
    print_usage()
    mtcnn_model_dir = '../model'
    save_dir = './megaface_json'

    fd_json_fn = './identity_json_list.json'
    
    print(sys.argv)

    if len(sys.argv) > 1:
        fd_json_fn = sys.argv[1]

    if len(sys.argv) > 2:
        mtcnn_model_dir = sys.argv[2]

    if len(sys.argv) > 3:
        save_dir = sys.argv[3]

    main(fd_json_fn, mtcnn_model_dir, save_dir)
