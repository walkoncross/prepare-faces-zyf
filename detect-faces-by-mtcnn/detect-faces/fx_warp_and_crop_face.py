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

# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
]

DEFAULT_CROP_SIZE = (96, 112)

class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
                __file__, super.__str__(self))

def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                output_padding=(0, 0),
                                output_square=False):
    """
    Function:
    ----------
        get reference 5 key points according to crop settings:

        1. If output_square: make the default crop_size (96, 112) into a square;
        2. Pad the crop_size by inner_padding_factor in each side;
        3. Resize crop_size into (output_size - output_padding*2),
            pad into output_size with output_padding;
        4. Output reference_5point;

    Parameters:
    ----------
        @output_size: (w, h) or None
            size of aligned face image
        @inner_padding_factor: (w_factor, h_factor)
            padding factor for inner (w, h)
        @output_padding: (w_pad, h_pad)
            each row is a pair of coordinates (x, y)
        @output_square: True or False
            if True:
                 make the default crop_size (96, 112) into a square before padding;
            else:
                keep the crop ratio in default crop_size (96, 112) before padding;

    Returns:
    ----------
        @reference_5point: 5x2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    if output_size is None:
        return REFERENCE_FACIAL_POINTS

    if not (output_padding[0] < output_size[0]
           and output_padding[1] < output_size[1]):
        raise FaceWarpException('Not (output_padding[0] < output_size[0]'
           'and output_padding[1] < output_size[1])')

    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    if (output_size == DEFAULT_CROP_SIZE
            and inner_padding_factor == 0
            and output_padding == (0, 0)
            and output_square is False):
        return REFERENCE_FACIAL_POINTS

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 1) make the inner region a square
    if output_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts = tmp_5pts + size_diff / 2
        tmp_crop_size = tmp_crop_size + size_diff

    # 2) pad the inner region according inner_padding_factor
    size_diff = tmp_crop_size * inner_padding_factor * 2
    tmp_5pts = tmp_5pts + size_diff / 2
    tmp_crop_size = tmp_crop_size + size_diff

    # 3) resize the padded inner region
    scale_factor = (np.array(output_size) -
                    np.array(output_padding) * 2
                    ) / tmp_crop_size
    tmp_5pts = tmp_5pts * scale_factor

    # 4) add output_padding
    reference_5point = tmp_5pts + np.array(output_padding)

    return reference_5point


def get_affine_transform_matrix(src_pts, dst_pts):
    """
    Function:
    ----------
        get affine transform matrix 'tfm' from src_pts to dst_pts

    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points matrix, each row is a pair of coordinates (x, y)

        @dst_pts: Kx2 np.array
            destination points matrix, each row is a pair of coordinates (x, y)

    Returns:
    ----------
        @tfm: 2x3 np.array
            transform matrix from src_pts to dst_pts
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


def warp_and_crop_face(src_img,
                       facial_pts,
                       reference_pts=None,
                       crop_size=(96, 112),
                       align_type='smilarity'):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv

    Parameters:
    ----------
        @src_img: 3x3 np.array
            input image
        @facial_pts: could be
            1)a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        @reference_pts: could be
            1) a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        or
            3) None
            if None, use default reference facial points
        @crop_size: (w, h)
            output face image size
        @align_type: transform type, could be one of
            1) 'similarity': use similarity transform
            2) 'cv2_affine': use the first 3 points to do affine transform,
                    by calling cv2.getAffineTransform()
            3) 'affine': use all points to do affine transform

    Returns:
    ----------
        @face_img: output face image with size (w, h) = @crop_size
    """

    if reference_pts is None:
        reference_pts = REFERENCE_FACIAL_POINTS

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException('reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException('facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

#    print '--->src_pts:\n', src_pts
#    print '--->ref_pts\n', ref_pts

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            'facial_pts and reference_pts must have the same shape')

    if align_type is 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
#        print('cv2.getAffineTransform() returns tfm=\n' + str(tfm))
    elif align_type is 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
#        print('get_affine_transform_matrix() returns tfm=\n' + str(tfm))
    else:
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
#        print('get_similarity_transform_for_cv2() returns tfm=\n' + str(tfm))

#    print '--->Transform matrix: '
#    print('type(tfm):' + str(type(tfm)))
#    print('tfm.dtype:' + str(tfm.dtype))
#    print tfm

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img


if __name__ == '__main__':
    print '\n=================================='
    print 'test get_reference_facial_points()'

    output_square = True
    inner_padding_factor = 0.25
    output_padding = (0, 0)
    output_size = (224, 224)

    reference_5pts = get_reference_facial_points(output_size,
                                                 inner_padding_factor,
                                                 output_padding,
                                                 output_square)

    print '--->reference_5pts:\n', reference_5pts

    try:
        import matplotlib.pyplot as plt
        dft_5pts = np.array(REFERENCE_FACIAL_POINTS)
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
        plt.scatter(reference_5pts[:, 0], reference_5pts[:, 1])
    except Exception as e:
        print 'Exception caught when trying to plot: ', e

    print '\n=================================='
    print 'test warp_and_crop_face()'

    img_fn = '../test_imgs/Jennifer_Aniston_0016.jpg'
    # imgSize = [96, 112]; # cropped dst image size

    # facial points in cropped dst image
    #    coord5points = [[30.2946, 65.5318, 48.0252, 33.5493, 62.7299],
    #                    [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]];

    # facial points in src image
    # facial5points = [[105.8306, 147.9323, 121.3533, 106.1169, 144.3622],
    #                 [109.8005, 112.5533, 139.1172, 155.6359, 156.3451]];
    facial5points = [[105.8306,  109.8005],
                     [147.9323,  112.5533],
                     [121.3533,  139.1172],
                     [106.1169,  155.6359],
                     [144.3622,  156.3451]
                     ]

#    facial5points = np.array(facial5points)
    print('Loading image {}'.format(img_fn))
    image = cv2.imread(img_fn, True)

    # for pt in src_pts[0:3]:
    #    cv2.circle(image, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), 1, 8, 0)

    # swap BGR to RGB to show image by pyplot
    image_show = image[..., ::-1]

    plt.figure()
    plt.title('src image')
    plt.imshow(image_show)

    def crop_test(image,
             facial5points,
             reference_points=None,
             output_size=(96, 112),
             align_type='similarity'):


        #dst_img = transform_and_crop_face(image, facial5points, coord5points, imgSize)

        dst_img = warp_and_crop_face(image,
                                     facial5points,
                                     reference_points,
                                     output_size,
                                     align_type)

        print 'warped image shape: ', dst_img.shape

        # swap BGR to RGB to show image by pyplot
        dst_img_show = dst_img[..., ::-1]

        plt.figure()
        plt.title(align_type + ' transform ' + str(output_size))
        plt.imshow(dst_img_show)

    print '===>test default crop setting with similarity transform'
    crop_test(image, facial5points)

    print '===>test default crop setting with cv2 affine transform'
    crop_test(image, facial5points, align_type='cv2_affine')

    print '===>test default crop setting with default affine transform'
    crop_test(image, facial5points, align_type='affine')

    print '===>test default square crop setting'
    # crop settings, set the region of cropped faces
    output_square = True
    inner_padding_factor = 0.25
    output_padding = (0, 0)
    output_size = (224, 224)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, output_padding, output_square)
    print '--->reference_5pts:\n', reference_5pts

    print '===>test default square crop setting with similarity transform'
    crop_test(image, facial5points, reference_5pts, output_size)

    print '===>test default square crop setting with similarity transform'
    crop_test(image, facial5points, reference_5pts, output_size,
              align_type='cv2_affine')

    print '===>test default crop setting with default affine transform'
    crop_test(image, facial5points, reference_5pts, output_size,
              align_type='affine')
