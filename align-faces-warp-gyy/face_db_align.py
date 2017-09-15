import numpy as np
#from mtcnn.demo import FindLandmark
import cv2


class AlignImage(object):
    """docstring for AlignImage"""

    def __init__(self):
        super(AlignImage, self).__init__()
        self.height = 128
        self.width = 128

    def changecoord(self, M, pts):
        MX = []
        MX.append(pts[: 5])
        MX.append(pts[5:])
        MX = np.array(MX)
        MX = np.vstack((MX, np.array([1, 1, 1, 1, 1])))
        MX = np.dot(M, MX)
        MX[0] = MX[0] / MX[2]
        MX[1] = MX[1] / MX[2]
        pts = []
        for i in range(5):
            pts.append(MX[0][i])
        for i in range(5):
            pts.append(MX[1][i])
        return pts

    def align(self, im, points):
        #points = FindLandmark(im)
        images = []
        for pts in points:
            Lefteye = (pts[0], pts[5])
            Righteye = (pts[1], pts[6])
            eyecs = ((Lefteye[0] + Righteye[0]) / 2,
                     (Lefteye[1] + Righteye[1]) / 2)
            cos_theta = (Righteye[0] - Lefteye[0]) / np.sqrt(
                (Righteye[0] - Lefteye[0])**2 + (Righteye[1] - Lefteye[1])**2)
            sin_theta = (Righteye[1] - Lefteye[1]) / np.sqrt(
                (Righteye[0] - Lefteye[0])**2 + (Righteye[1] - Lefteye[1])**2)
            M1 = np.float32(
                [[1, 0, -eyecs[0]], [0, 1, -eyecs[1]], [0, 0, 1]])  # shift1
            M2 = np.float32([[cos_theta, sin_theta, 0],
                             [-sin_theta, cos_theta, 0], [0, 0, 1]])  # rotate
            M3 = np.float32(
                [[1, 0, eyecs[0]], [0, 1, eyecs[1]], [0, 0, 1]])  # shift2
            M = np.dot(M3, np.dot(M2, M1))
            pts = self.changecoord(M, pts)
            Lefteye = (pts[0], pts[5])
            Righteye = (pts[1], pts[6])
            Leftmouth = (pts[3], pts[8])
            Rightmouth = (pts[4], pts[9])
            img = cv2.warpPerspective(im, M, (im.shape[1], im.shape[0]))
            mouthcs = ((Leftmouth[0] + Rightmouth[0]) / 2,
                       (Leftmouth[1] + Rightmouth[1]) / 2)
            eyecs = ((Lefteye[0] + Righteye[0]) / 2,
                     (Lefteye[1] + Righteye[1]) / 2)
            ec_mc_y = np.sqrt((eyecs[0] - mouthcs[0]) **
                              2 + (eyecs[1] - mouthcs[1])**2)
            scaling = 48 * 1.0 / ec_mc_y
            ec_y = 40
            M4 = np.float32([[scaling, 0, 0], [0, scaling, 0], [0, 0, 1]])
            img = cv2.warpPerspective(
                img, M4, (int(im.shape[1] * scaling), int(im.shape[0] * scaling)))
            pts = self.changecoord(M4, pts)
            Lefteye = (pts[0], pts[5])
            Righteye = (pts[1], pts[6])
            Nose = (pts[2], pts[7])
            Leftmouth = (pts[3], pts[8])
            Rightmouth = (pts[4], pts[9])
            eyecs = ((Lefteye[0] + Righteye[0]) / 2,
                     (Lefteye[1] + Righteye[1]) / 2)
            shiftx = 64 - eyecs[0]
            shifty = 40 - eyecs[1]
            M5 = np.float32(
                [[1, 0, shiftx], [0, 1, shifty], [0, 0, 1]])  # shift3
            img = cv2.warpPerspective(
                img, M5, (self.width, self.height))
            pts = self.changecoord(M5, pts)
            images.append(img)
        return images

if __name__ == '__main__':
    AlignImage = AlignImage()
    im = cv2.imread("/home/zhouzhao/Projects/FACE_VERFICATION/face_verfication/datatest/6.jpg")
    images = AlignImage.align(im)
    for im in images:
        cv2.imshow("im",im)
        cv2.waitKey(0)
