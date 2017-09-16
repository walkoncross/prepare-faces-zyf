import numpy as np
import sys
import cv2
import json
import os
import os.path as osp
from skimage import io
import urllib
from mtcnn_aligner import MtcnnAligner

def main(mtcnn_model_dir, save_dir=None, save_img=None):
    if save_dir is None:
        save_dir = './gender_origin/label'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)
	
    if save_img is None:
        save_img = './gender_origin/img'

    if not osp.exists(save_img):
        os.makedirs(save_img)

    aligner = MtcnnAligner(mtcnn_model_dir, False)

    fp = open('/disk2/du/mtcnn-caffe-good/mtcnn_aligner/dataset-gender-all.json','r')
    all_lines = fp.readlines()
    count = 23097
    for line in all_lines[23097:]:
        count = count + 1
        print count
        data = json.loads(line)

        image_url = data["url"]
        image_bbox = data["label"]["detect"]["general_d"]["bbox"]
        file_idx = "%06d" % count
        print file_idx
        filename = str(file_idx)+'.json'
        imgname = str(file_idx)+'.jpg'
        urllib.urlretrieve(image_url,osp.join(save_img,imgname))
        print('===> Save image: ' + filename) 
	
        fp_rlt = open(osp.join(save_dir,filename),'w')
        item = {}
        item['url'] = image_url
        item['imgname'] = imgname
	
        if image_bbox == []:
            item['detect'] = []
            #continue
        else:
     	    img = io.imread(image_url)
            if len(img.shape)!=3:
                item['detect'] = []
                #continue
            else:
                item['detect'] = []
 	        for idx in range(len(image_bbox)):
                    print "len of box",len(image_bbox)
	            #item['detect'] = []
                    per_bbox=image_bbox[idx]["pts"]
                    GT_RECT = [per_bbox[0][0],per_bbox[0][1],per_bbox[2][0],per_bbox[2][1]]
				
	            boxes, points = aligner.align_face(img, [GT_RECT])
                    pts = points[0]
            
                    label=image_bbox[idx]["class"]
                    tmp = {'rect': GT_RECT,'pts': pts,'class': label}
                    item['detect'].append(tmp)
				
        print "item:",item
        json.dump(item, fp_rlt, indent=4)
        fp_rlt.close()
	
if __name__ == "__main__":
    #print_usage()
    mtcnn_model_dir = '../model'
    save_dir = None
    save_img = None

    print(sys.argv)

    if len(sys.argv) > 1:
        mtcnn_model_dir = sys.argv[1]

    if len(sys.argv) > 2:
        save_dir = sys.argv[2]

    if len(sys.argv) > 3:
        save_img = sys.argv[3]

    main(mtcnn_model_dir, save_dir, save_img)	
