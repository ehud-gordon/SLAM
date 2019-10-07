import numpy as np
import cv2
import os
from PARAMS import *


def undistort_image(im, K, dist):
    """
    :param im: ndarray of image
    :param K: 3x3 ndarray,  Camera Matrix,
    :param dist:  ndarray distortion coefficients
    :return: ndarray of undistorted image
    """
    # Get optimal camera matrix for better undistortion
    w,h = im.shape[0:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    # Undistort images
    im_undistorted = cv2.undistort(im, K, dist, newCameraMatrix=new_camera_matrix)
    # crop and save the image
    x, y, w, h = roi
    im_undistorted = im_undistorted[y:y + h, x:x + w]
    return im_undistorted


def main():
    for frame_name in os.listdir(os.path.join(ROOT_PATH, FRAMES_DISTORTED_PATH)):
        im = cv2.imread(os.path.join(ROOT_PATH, FRAMES_DISTORTED_PATH, frame_name), cv2.IMREAD_UNCHANGED)
        undistorted_im = undistort_image(im, CAMERA_MATRIX, DIST_COEF)
        cv2.imwrite(os.path.join(ROOT_PATH, FRAMES_PATH, frame_name), undistorted_im)
        print('undistortrd', frame_name)

if __name__ == '__main__':
    main()