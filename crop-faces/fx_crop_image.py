# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:20:43 2017

@author: zhaoy
"""

import numpy as np

def crop_square_image_into_ratio(src_img, w_ratio, h_ratio):
    assert( src_img.ndim==3 and
            src_img.shape[2]==3 and
            src_img.shape[0]==src_img.shape[1] and
            w_ratio>0 and h_ratio>0
        )

    img_sz = src_img.shape[0]
    if w_ratio==h_ratio:
        return src_img
    elif w_ratio > h_ratio:
        new_ht = int(float(img_sz) * h_ratio / w_ratio)
        crop_sz = (img_sz - new_ht) / 2
        dst_img = src_img[crop_sz:img_sz-crop_sz, ...]
        return dst_img
    else:
        new_wd = int(float(img_sz) * w_ratio / h_ratio)
        crop_sz = (img_sz - new_wd) / 2
        dst_img = src_img[:, crop_sz:img_sz-crop_sz, ...]
        return dst_img