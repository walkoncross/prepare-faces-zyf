import numpy as np
import cv2
import json
import os
import os.path as osp
import sys


def main(img_list_file, fd_json):
    fp_json = open(fd_json, 'r')
    fd_list = json.load(fp_json)
    fp_json.close()
       
    fp = open(img_list_file,'r')
    
    fn_rlt =  osp.splitext(fd_json)[0] + '_fixed.json'
    fp_rlt = open(fn_rlt, 'w')

    i = 0
    for line in fp:
        print line
        line_split = line.split()
        
        img_fn = line_split[0]
        id_num = line_split[1]
        
        print 'process image: ', img_fn, " id_num: ", id_num

        fd_list[i]['filename'] = img_fn

    json.dump(fd_list, fp_rlt, indent=2)
    fp_rlt.close()
    fp.close()
    


if __name__ == "__main__":
    
    fd_json = '/disk2/data/FACE/idcard_dateset_mtcnn_aligned/fd_rlt.json'

    img_list_fn = '/disk2/zhaoyafei/facex-sim-test/img_list_all-shortname.txt'

    main(img_list_fn, fd_json)
