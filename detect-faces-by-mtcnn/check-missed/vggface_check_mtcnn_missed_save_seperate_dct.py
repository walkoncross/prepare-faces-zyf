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

overlap_thresh = 0.3

log_fn1 = 'fd_succeeded_list.txt'
log_fn2 = 'fd_missed_list.txt'


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

def main(face_json_file, file_root_path, mtcnn_model_dir, save_dir=None):
    if save_dir is None:
        save_dir = './fd_json_add_missed'

    if not osp.exists(save_dir):
        print('mkdir for aligned faces, aligned root dir: ', save_dir)
        os.makedirs(save_dir)

    aligner = MtcnnAligner(mtcnn_model_dir, False)

    fp_log1 = open(osp.join(save_dir, log_fn1), 'w')
    fp_log2 = open(osp.join(save_dir, log_fn2), 'w')

    missed_count1 = 0
    missed_count2 = 0
    
    fp = open('../mtcnn_detector/gender-file/dataset-31-1501643272350-gender-7.json','r')
    all_lines = fp.readlines()
    count = 1
    for line in all_lines:
        err_msg = ''
        print count
        count = count + 1
        data = json.loads(line)

        image_url = data["url"]        

        if 'filename' not in item:
            err_msg = "'filename' not in item, break..."
            print(err_msg)
            fp_log2.write(err_msg + '\n')
            break

        overlap_thresh_0 = overlap_thresh
       
        print osp.join(save_dir,'result_after_miss',data[u'url'].split('/')[-2])
        isExists1 = osp.exists(osp.join(save_dir,'result_after_miss',data[u'url'].split('/')[-2]))
        if not isExists1:
            os.makedirs(osp.join(save_dir,'result_after_miss',data[u'url'].split('/')[-2]))
        isExists2 = os.path.exists(osp.join(save_dir,'result_after_miss',filename))
        #if isExists2:
        #    print(filename + "has alreadly processed")
        #    continue


        fp_rlt = open(osp.join(save_dir,'result_after_miss',filename), 'w')

        print('===> Processing image: ' + filename)

        if 'faces' not in item:
            err_msg = "'faces' not in item"
            fp_log2.write(item['filename'] + ': ' + err_msg + '\n')
            continue
        elif 'face_count' not in item:
            err_msg = "'face_count' not in item"
            fp_log2.write(item['filename'] + ': ' + err_msg + '\n')
            continue

        if item['message'] != 'success':
            err_msg = "message fail to load"
            fp_log2.write(item['filename'] + ': ' + err_msg + '\n')
            json.dump(item, fp_rlt, indent=4)
            fp_rlt.close()
            continue

        nfaces = item['face_count']
        GT_RECT = [item['gt'][0][0],item['gt'][0][1],item['gt'][2][0],item['gt'][2][1]]

        if nfaces < 1 :
            missed_count1 += 1
            fp_log2.write(item['filename'] + ': ' +
                          "item['face_count'] < 1" + '\n')

            #img = cv2.imread(img_fn)
            img = io.imread(data[u'url'])
            boxes, points = aligner.align_face(img, [GT_RECT])

            item['face_count'] = 1
            box = boxes[0]
            pts = points[0]

            tmp = {'rect': box[0:4],
                   'score': box[4],
                   'pts': pts
                   }
            item['faces'].append(tmp)
            item['message'] = 'success'
            item['used_gt'] = 1

            fp_log2.write('-->faces added by aligner:\n{}\n'.format(tmp))
            json.dump(item, fp_rlt, indent=4)
            fp_rlt.close()
            continue
	print 'gt',item['gt'][0][0],',',item['gt'][0][1],',',item['gt'][2][0],',',item['gt'][2][1]
        #GT_RECT = [item['gt'][0][0],item['gt'][0][1],item['gt'][2][0],item['gt'][2][1]]
        overlaps = get_gt_overlap(item['faces'],GT_RECT)

        max_overlap_idx = overlaps.argmax()

        if overlaps[max_overlap_idx] >= overlap_thresh_0:
            fp_log1.write(item['filename'] + ': ' + " max_overlap_idx="
                          + str(max_overlap_idx) + '\n')
        else:
            missed_count2 += 1

            fp_log2.write(item['filename'] + ': ' +
                          "no faces have overlap>={} with groundtruth".format(
                              overlap_thresh_0) +
                          '\n')
            fp_log2.write("--> max_overlap_idx = {}\n".format(max_overlap_idx))
            fp_log2.write("--> overlaps = {}\n".format(overlaps))

            #img = cv2.imread(img_fn)
            img = io.imread(data[u'url'])
            boxes, points = aligner.align_face(img, [GT_RECT])

            item['face_count'] += 1
            box = boxes[0]
            pts = points[0]

            tmp = {'rect': box[0:4],
                   'score': box[4],
                   'pts': pts
                   }
            item['faces'].append(tmp)
            item['used_gt'] = 1
            fp_log2.write('-->faces added by aligner:\n{}\n'.format(tmp))
        json.dump(item, fp_rlt, indent=4)
        fp_rlt.close()

    fp_log2.write("\n==>Images with missed faces: {}\n".format(missed_count1
                                                   + missed_count2))
    fp_log2.write("\t{} missed because of no detection\n".format(missed_count1))
    fp_log2.write("\t{} missed because of max_overlap<thresh\n".format(
        missed_count2))

    #json.dump(img_list, fp_rlt, indent=4)
    #fp_rlt.close()

    fp_log1.close()
    fp_log2.close()

if __name__ == "__main__":
    print_usage()
    mtcnn_model_dir = '../model'
    save_dir = None

    #fd_json_fn = r'../../prepare-faces-zyf/lfw-mtcnn-fd-rlt/lfw-mtcnn-v2-matlab-fd-rlt-3imgs.json'
    fd_json_fn = '../mtcnn_detector/vggface_mtcnn_test.json'
    #fd_json_fn = r'../../prepare-faces-zyf/lfw-mtcnn-fd-rlt/lfw_mtcnn_fd_rlt.json'
    #img_root_dir = '../mtcnn_detector/'
    file_root_path = '../mtcnn_detector/result_json'

    #fd_json_fn = r'../../lfw-mtcnn-fd-rlt/lfw-mtcnn-v2-matlab-fd-rlt-3imgs.json'
    #fd_json_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_falied3_align_rlt.json'
    #fd_json_fn = r'../../lfw-mtcnn-fd-rlt/lfw_mtcnn_fd_rlt_kirk.json'
    #img_root_dir = r'/disk2/data/FACE/LFW/LFW'

    print(sys.argv)

    if len(sys.argv) > 1:
        fd_json_fn = sys.argv[1]

    if len(sys.argv) > 2:
        #img_root_dir = sys.argv[2]
        file_root_path = sys.argv[2]

    if len(sys.argv) > 3:
        mtcnn_model_dir = sys.argv[3]

    if len(sys.argv) > 4:
        save_dir = sys.argv[4]

    main(fd_json_fn, file_root_path, mtcnn_model_dir, save_dir)
