import numpy as np


# PARAMS
ROOT_PATH = "C:\\Users\\idob\\Documents\\huji\\vision\\project\\root_dir_tik"
FRAMES_DISTORTED_PATH = "frames_distorted"
FRAMES_PATH = "frames"
POINTS_PATH = "points"
MATCHES_PATH = "matches"
RTS_NAME = "RTs"
START_FRAME = 0
FRAMES_STEP = 10
NUM_OF_SIFT_POINTS = 2000
NUM_OF_KF_TO_FOLLOW = 2
SIFT_DESC_RATIO = 0.5
RANSAC_THRESHOLD = 0.0005
CAMERA_MATRIX = np.eye(3)
EXAMPLES_FOR_ETA = 7
VERBOSE = False
# CAMERA_MATRIX = np.array([[275.90391284,   0.        , 172.30768348],
#                           [  0.        , 251.75705344, 116.92293755],
#                           [  0.        ,   0.        ,   1.        ]])
# DIST_COEF = np.array([[ 2.33443433e-01, -1.20387973e+00, -4.53533007e-04,
#                         -4.78190962e-03,  1.81872558e+00]])

# CAMERA_MATRIX = np.array([[521.59247169,   0.        , 315.55785298],
#                           [  0.        , 522.75312436, 232.42967732],
#                           [  0.        ,   0.        ,   1.        ]])
# DIST_COEF = np.array([[ 2.42105525e-01, -1.28814427e+00, -7.97625937e-04,
#                         2.34116288e-03,  2.11093224e+00]])


# CAMERA_MATRIX = np.array([[3.23645613e+03, 0.00000000e+00, 2.01195800e+03],
#                           [0.00000000e+00, 3.24835473e+03, 1.49468063e+03],
#                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# DIST_COEF = np.array([[ 2.42105525e-01, -1.28814427e+00, -7.97625937e-04,
#                         2.34116288e-03,  2.11093224e+00]])


CAMERA_MATRIX = np.array([[517.99912642,   0.        , 306.77789797],
                          [  0.        , 519.76289458, 224.82133308],
                          [  0.        ,   0.        ,   1.        ]])
DIST_COEF = np.array([[ 1.78317842e-01, -6.44898180e-01, -6.30748991e-04,
                        7.06162090e-04,  4.13288621e-01]])