# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 00:57:38 2016

@author: zhaoy
"""

import numpy as np


def get_image_roi(image, roi, roi_scale=1.0, output_square=0):
    '''
    Get a sub-image from ROI of an image

    Parameters
    ----------
    image: input image, type: numpy.mdarray
    roi: input ROI, in form of a list [x, y, w, h, ...] or a tuple (x, y, w, h, ...)
    roi_scale: the factor to scale the ROI before crop
    output_square: if set 1, output squared images

    Return: an image, type: numpy.mdarray
    '''

    if None is image:
        return None

    cx = int(roi[0] + roi[2] / 2.0)
    cy = int(roi[1] + roi[3] / 2.0)
    wd = int(roi[2] * roi_scale)
    ht = int(roi[3] * roi_scale)

    if output_square:
        wd = max(wd, ht)
        ht = wd

    roi_img = np.zeros((ht, wd, image.shape[2]), image.dtype)

    x0 = cx - wd / 2
    y0 = cy - ht / 2

    x1 = cx + wd / 2
    y1 = cy + ht / 2

    dx0 = 0
    dy0 = 0

    if x0 < 0:
        dx0 = -x0
        x0 = 0

    if y0 < 0:
        dy0 = -y0
        y0 = 0

    x1 = min(image.shape[1] - 1, x1)
    y1 = min(image.shape[0] - 1, y1)

    wd = x1 - x0
    ht = y1 - y0

    roi_img[dy0:dy0 + ht, dx0:dx0 + wd] = image[y0:y1, x0:x1]
    return roi_img


def get_image_roi_list(image, roi_list, roi_scale=1.0, output_square=0):
    '''
    Get a list of sub-images from ROIs of an image

    Parameters
    ----------
    image: input image, type: numpy.mdarray
    roi_list: list or tuple of input ROIs,
            each ROI is in form of a list [x, y, w, h, ...] or a tuple (x, y, w, h, ...)
    roi_scale: the factor to scale the ROI before crop
    output_square: if set 1, output squared images

    Return: list of images, image type: numpy.mdarray
    '''
    if None is image:
        return None

    roi_img_list = []
    for roi in roi_list:
        roi_img = get_image_roi(image, roi, roi_scale, output_square)
        roi_img_list.append(roi_img)

    return roi_img_list


def get_image_roi_by_4pts(image, roi, roi_scale=1.0, output_square=0):
    '''
    Get a sub-image from ROI of an image

    Parameters
    ----------
    image: input image, type: numpy.mdarray
    roi: input ROI, in form of a list [[xl, yt],[xr, yt], [xr, yb], [xl, yb], ...] 
        or a tuple ([xl, yt],[xr, yt], [xr, yb], [xl, yb], ...)
    roi_scale: the factor to scale the ROI before crop
    output_square: if set 1, output squared images

    Return: an image, type: numpy.mdarray
    '''
    _roi =[ roi[0][0], roi[0][1], roi[2][0] - roi[0][0], roi[2][1] - roi[0][1]]

    roi_img = get_image_roi(image, _roi, roi_scale, output_square)

    return roi_img


def get_image_roi_list_by_4pts(image, roi_list, roi_scale=1.0, output_square=0):
    '''
    Get a list of sub-images from ROIs of an image

    Parameters
    ----------
    image: input image, type: numpy.mdarray
    roi_list: list or tuple of input ROIs,
            each ROI is in form of a list [[xl, yt],[xr, yt], [xr, yb], [xl, yb], ...] 
            or a tuple ([xl, yt],[xr, yt], [xr, yb], [xl, yb], ...)
    roi_scale: the factor to scale the ROI before crop
    output_square: if set 1, output squared images

    Return: list of images, image type: numpy.mdarray
    '''
    if None is image:
        return None

    roi_img_list = []
    for roi in roi_list:
        _roi =[ roi[0][0], roi[0][1], roi[2][0] - roi[0][0], roi[2][1] - roi[0][1]]

        roi_img = get_image_roi(image, _roi, roi_scale, output_square)
        roi_img_list.append(roi_img)

    return roi_img_list
