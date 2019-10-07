import numpy as np
import cv2
import os
from utils import *

# def get_cur_dir():
#     realpath = os.path.realpath(__file__)
#     curdir = os.path.dirname(realpath)
#     return curdir
#
# def display_image(im, title=""):
#     plt.figure()
#     plt.imshow(im, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

n1 = 0
n2 = 10
def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    img1 = img1.copy()
    img2 = img2.copy()
    r, c = img1.shape[:2]
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    display_image(img1[:,:,::-1], title="drawLines before im1")
    display_image(img2[:,:,::-1], title="drawLines before im2")
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    display_image(img1[:, :, ::-1], title="drawLines after im1")
    display_image(img2[:, :, ::-1], title="drawLines after im2")
    return img1, img2

def drawEpipole(image1, image2):
    """
    ransacReprojThreshold:	Parameter used only for RANSAC. It is the maximum distance from a point to an epipolar line in pixels,
      beyond which the point is considered an outlier and is not used for computing the final fundamental matrix. It can be set to something like 1-3,
      depending on the accuracy of the point localization, image resolution, and the image noise.
    confidence:	Parameter used for the RANSAC and LMedS methods only. It specifies a desirable level of confidence (probability) that the estimated matrix is correct.
    :param image1: ndarray
    :param image2:
    :return:
    """

    RT, matches, pts_coord = read_matches(n1, n2)
    sift_points1, _, _ = read_points(n1)
    sift_points1 = sift_points1[matches[:, 0]]
    sift_points2, _, _ = read_points(n2)
    sift_points2 = sift_points2[matches[:, 1]]
    # pts1 = matches[:,:2]
    # pts2 = matches[:,2:4]
    pts1 = np.int32(sift_points1)
    pts2 = np.int32(sift_points2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.RANSAC,ransacReprojThreshold=1., confidence=0.99)
    # get essential Matrix
    # curdir = get_cur_dir()
    # K = np.load(os.path.join(curdir, 'camera_params_new_480', 'K.npy'))
    # E,mask_e = cv2.findEssentialMat(pts1, pts2, K, prob=0.99, threshold=1.)
    # E2 = E / E[2,2]
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(image1, image2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(image2, image1, lines2, pts2, pts1)
    # plt.subplot(121), plt.imshow(img5[:,:,::-1])
    # plt.subplot(122), plt.imshow(img3[:,:,::-1])
    # plt.show()


def main():
    curdir = get_cur_dir()
    rectification = False
    # without rectification
    imgs_dir = os.path.join(curdir, 'frames')

    FILE_NAME1 = os.path.join(imgs_dir, 'frame{}.png'.format(n1))
    FILE_NAME2 = os.path.join(imgs_dir, 'frame{}.png'.format(n2))
    image1 = cv2.imread(FILE_NAME1, 0)
    image2 = cv2.imread(FILE_NAME2, 0)
    # resize
    drawEpipole(image1, image2)

main()
