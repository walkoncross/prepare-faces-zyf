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
    usage = 'python %s <img-list-file> <img-root-dir> <MTCNN-model-dir> [<save-dir>]' % osp.basename(__file__)
    print('USAGE: ' + usage)

GT_RECT = [0,0,255,255]
  
overlap_thresh = 0.3

output_size = (96, 112)
reference_5pts = None

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

        
def main(img_list_file, root_dir, mtcnn_model_dir, save_dir=None):
    if not save_dir:
        save_dir = './aligned_images'
        
    if not osp.exists(save_dir):
        print('mkdir for aligned faces, aligned root dir: ', save_dir)
        os.makedirs(save_dir)

    aligned_save_dir = osp.join(save_dir, 'aligned_faces')
    if not osp.exists(aligned_save_dir):
        print('mkdir for aligned faces, aligned images dir: ', aligned_save_dir)
        os.makedirs(aligned_save_dir)
        
    aligner = MtcnnAligner(mtcnn_model_dir, False)

    fp = open(img_list_file,'r')
    
    fn_rlt =  osp.join(save_dir, 'fd_rlt.json')
    fp_rlt = open(fn_rlt, 'w')
    fp_rlt.write('[\n')

    count = 0
    for line in fp:
        print line
        line_split = line.split()
        
        img_fn = line_split[0]
        id_num = line_split[1]
        
        img_fn_split = img_fn.split('/')
        
        img_full_fn = osp.join(root_dir, img_fn)
        
        print 'process image: ', img_full_fn, " id_num: ", id_num
    #for root,dirs,files in path_walk:
        err_msg = ''

        if not count:
            fp_rlt.write(',\n')
           
        count = count + 1
        print 'count: ', count

        overlap_thresh_0 = overlap_thresh

        save_subdir = osp.join(aligned_save_dir, img_fn_split[-2])
        save_img_fn = osp.join(save_subdir, img_fn_split[-1])
               
        if not osp.exists(save_subdir):
            os.makedirs(save_subdir)

        image = cv2.imread(img_full_fn)

        print image.shape
        boxes, points = aligner.align_face(image, [GT_RECT])

        box = boxes[0]
        pts = points[0]

        facial5points = np.reshape(points, (2, -1))
        dst_img = warp_and_crop_face(image, facial5points, reference_5pts, output_size)
        cv2.imwrite(save_img_fn, dst_img)

        item = {
                'filename': img_base_fn,
                'face_count': 1,
               }
        tmp = {'rect': box[0:4],
               'score': box[4],
               'pts': pts,
               'id': id_num
              }
        item['faces'] = [tmp]
        
        #item['id'] = data[u'url'].splitit('/')[-3]
        item['shape'] = image.shape
        json_str = json.dumps(item, indent=2)
        
        fp_rlt.write(json_str+'\n')
        fp_rlt.flush()
    
    fp_rlt.write(']\n')
    fp_rlt.close()
    fp.close()
    


if __name__ == "__main__":
    print_usage()
    mtcnn_model_dir = '../model'
    
    save_dir = '/disk2/data/FACE/idcard_dateset_mtcnn_aligned'

    img_list_fn = '../../facex-sim-test/identities-dataset_list.txt'
    root_dir = '/disk2/data/FACE/identities-dataset/assets/face_card_data'
    
    print(sys.argv)

    if len(sys.argv) > 1:
        img_list_fn = sys.argv[1]

    if len(sys.argv) > 2:
        root_dir = sys.argv[2]
        
    if len(sys.argv) > 3:
        mtcnn_model_dir = sys.argv[3]

    if len(sys.argv) > 4:
        save_dir = sys.argv[4]

    main(img_list_fn, root_dir, mtcnn_model_dir, save_dir)
