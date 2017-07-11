# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:43:29 2017

@author: zhaoy
"""
import numpy as np
import cv2

# from scipy.linalg import lstsq
# from scipy.ndimage import geometric_transform  # , map_coordinates

from matlab_cp2tform import get_similarity_transform_for_cv2

dft_normalized_5points = [
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
]

dft_crop_size = (96, 112)


def get_normalized_5points(output_size, padding_factor=0.0,
                           output_padding=(0, 0), output_square=False):
    '''
    1. If output_square: make the default_crop_size into a square;
    2. Pad the crop_size by padding_factor in each side;
    3. Resize crop_size into (output_size - output_padding*2),
        pad into output_size with output_padding;
    4. Output normalized_5point;
    '''
    assert(0 <= padding_factor <= 1.0)

    if output_size == dft_crop_size \
            and padding_factor == 0 \
            and output_padding == (0, 0) \
            and output_square is False:
        return dft_normalized_5points

    tmp_5pts = np.array(dft_normalized_5points)
    tmp_crop_size = np.array(dft_crop_size)

    if output_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts = tmp_5pts + size_diff / 2
        tmp_crop_size = tmp_crop_size + size_diff

    size_diff = tmp_crop_size * padding_factor * 2
    tmp_5pts = tmp_5pts + size_diff / 2
    tmp_crop_size = tmp_crop_size + size_diff

    scale_factor = (np.array(output_size) -
                    np.array(output_padding) * 2) / tmp_crop_size
    tmp_5pts = tmp_5pts * scale_factor

    tmp_5pts = tmp_5pts + np.array(output_padding)

    return tmp_5pts


def _get_transform_matrix(src_pts, dst_pts):
    """

    """
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

#    print('src_pts_:\n' + str(src_pts_))
#    print('dst_pts_:\n' + str(dst_pts_))

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

#    print('np.linalg.lstsq return A: \n' + str(A))
#    print('np.linalg.lstsq return res: \n' + str(res))
#    print('np.linalg.lstsq return rank: \n' + str(rank))
#    print('np.linalg.lstsq return s: \n' + str(s))

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm


def warp_and_crop_face(src_img, facial_pts,
                            normalized_pts=None,
                            crop_size=dft_crop_size):
    if normalized_pts is None:
        normalized_pts = dft_normalized_5points

    pts_dst = np.float32(normalized_pts)
    if pts_dst.shape[0] == 2:
        pts_dst = pts_dst.T

    pts_src = np.float32(facial_pts)
    if pts_src.shape[0] == 2:
        pts_src = pts_src.T

#    tfm = cv2.getAffineTransform(pts_src[0:3], pts_dst[0:3])
#    print('cv2.getAffineTransform returns tfm=\n' + str(tfm))
#    print('type(tfm):' + str(type(tfm)))
#    print('tfm.dtype:' + str(tfm.dtype))

#    tfm = _get_transform_matrix(pts_src, pts_dst)
#    print('_get_transform_matrix returns tfm=\n' + str(tfm))
#    print('type(tfm):' + str(type(tfm)))
#    print('tfm.dtype:' + str(tfm.dtype))


    tfm = get_similarity_transform_for_cv2(pts_src, pts_dst)
#    print('_get_transform_matrix returns tfm=\n' + str(tfm))
#    print('type(tfm):' + str(type(tfm)))
#    print('tfm.dtype:' + str(tfm.dtype))

    print '--->Transform matrix: '
    print tfm

    dst_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return dst_img


if __name__ == '__main__':

    output_square = True
    padding_factor = 0.25
    output_padding = (0, 0)
    output_size = (224, 224)

    normalized_5pts = get_normalized_5points(
        output_size, padding_factor, output_padding, output_square)

    print normalized_5pts

    try:
        import matplotlib.pyplot as plt
        dft_5pts = np.array(dft_normalized_5points)
        plt.title('Default 5 pts')
#        plt.axis('equal')
        plt.axis([0, 96, 112, 0])
#        plt.xlim(0, 96)
#        plt.ylim(0, 112)
        plt.scatter(dft_5pts[:, 0], dft_5pts[:, 1])

        plt.figure()
        plt.title('Transformed new 5 pts')
#        plt.axis('equal')
        plt.axis([0, 224, 224, 0])
#        plt.xlim(0, 224)
#        plt.ylim(0, 224)
        plt.scatter(normalized_5pts[:, 0], normalized_5pts[:, 1])
    except Exception as e:
        print 'Exception caught when trying to plot: ', e

    img_fn = '../test_imgs/Jennifer_Aniston_0016.jpg'
    #imgSize = [96, 112]; # cropped dst image size

    # facial points in cropped dst image
    #    coord5points = [[30.2946, 65.5318, 48.0252, 33.5493, 62.7299],
    #                    [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]];

    # facial points in src image
    #facial5points = [[105.8306, 147.9323, 121.3533, 106.1169, 144.3622],
    #                 [109.8005, 112.5533, 139.1172, 155.6359, 156.3451]];
    facial5points = [[ 105.8306,  109.8005],
           [ 147.9323,  112.5533],
           [ 121.3533,  139.1172],
           [ 106.1169,  155.6359],
           [ 144.3622,  156.3451]
           ];

    def test(img_fn, facial5points,
             normalized_facial_points=None,
             output_size=(96,112)):
        print('Loading image {}'.format(img_fn))
        image = cv2.imread(img_fn, True);

        #for pt in pts_src[0:3]:
        #    cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), 1, 8, 0)

        image_show = image[...,::-1]# swap BGR to RGB to show image by pyplot

        plt.figure();
        plt.imshow(image_show)
        #dst_img = transform_and_crop_face(image, facial5points, coord5points, imgSize)

        dst_img = warp_and_crop_face(image, facial5points,
                                     normalized_facial_points,
                                     output_size)
        print 'warped image shape: ', dst_img.shape

        dst_img_show = dst_img[...,::-1]# swap BGR to RGB to show image by pyplot

        plt.figure()
        plt.imshow(dst_img_show )

    test(img_fn, facial5points)

    # crop settings, set the region of cropped faces
    output_square = True
    padding_factor = 0.25
    output_padding = (0, 0)
    output_size = (224, 224)

    # get the normalized 5 landmarks position in the crop settings
    normalized_5pts = get_normalized_5points(
        output_size, padding_factor, output_padding, output_square)
    print normalized_5pts

    test(img_fn, facial5points, normalized_5pts, output_size)

