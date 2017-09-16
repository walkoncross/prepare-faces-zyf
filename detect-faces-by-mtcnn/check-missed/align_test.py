# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 07:35:55 2017

@author: zhaoy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 06:33:36 2017

@author: zhaoy
"""
import sys
import os
import os.path as osp

import json
import time
import cv2

from mtcnn_aligner import MtcnnAligner, draw_faces


def main(img_path, face_rects, save_dir=None, save_img=True, show_img=True):
    if save_dir is None:
        save_dir = './fa_test_rlt'

    save_json = 'mtcnn_align_rlt.json'
    caffe_model_path = "../model"

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_rlt = open(osp.join(save_dir, save_json), 'w')
    results = []

    print '===> Processing image: ' + img_path

    img = cv2.imread(img_path)

    aligner = MtcnnAligner(caffe_model_path, False)

    rlt = {}
    rlt["filename"] = img_path
    rlt["faces"] = []
    rlt['face_count'] = 0

    t1 = time.clock()
    bboxes, points = aligner.align_face(img, face_rects)
    t2 = time.clock()

    n_boxes = len(face_rects)
    print("-->Alignment cost %f seconds, processed %d face rects, avg time: %f seconds" %
          ((t2 - t1), n_boxes, (t2 - t1) / n_boxes))

    if bboxes is not None and len(bboxes) > 0:
        for (box, pts) in zip(bboxes, points):
            #                box = box.tolist()
            #                pts = pts.tolist()
            tmp = {'rect': box[0:4],
                   'score': box[4],
                   'pts': pts
                   }
            rlt['faces'].append(tmp)

    rlt['face_count'] = len(bboxes)

    rlt['message'] = 'success'
    results.append(rlt)

    if save_img or show_img:
        draw_faces(img, bboxes, points)

    if save_img:
        save_name = osp.join(save_dir, osp.basename(img_path))
        cv2.imwrite(save_name, img)

    if show_img:
        cv2.imshow('img', img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    json.dump(results, fp_rlt, indent=4)
    fp_rlt.close()

if __name__ == '__main__':
    img_path = '../test_imgs/Marilyn_Monroe_0002.jpg'
    face_rects = [
        [
            [
                91,
                57
            ],
            [
                173,
                57
            ],
            [
                173,
                180
            ],
            [
                91,
                180
            ]
        ]
    ]

    save_dir = None

    main(img_path, face_rects, save_dir, save_img=True, show_img=True)
