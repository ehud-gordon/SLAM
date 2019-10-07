import numpy as np
from matplotlib import pyplot as plt
import math
import pickle
import os
import cv2
import time
from scipy.spatial.transform import Rotation
from PARAMS import *


#######################
#      File Names     #
#######################
def is_frame_name(name):
    return len(name) > 9 and name[:5] == 'frame' and name[-4:] == '.png' and name[5:-4].isdigit()


def is_points_name(name):
    return len(name) > 13 and name[:6] == 'points' and name[-7:] == '.pickle' and name[6:-7].isdigit()


def is_match_name(name):
    return len(name) >= 18 and name[:8] == 'matches_' and name[-7:] == '.pickle' and name[8:-7].count('_') and name[8:-7].split('_')[0].isdigit() and name[8:-7].split('_')[1].isdigit()


def name_to_i(name):
    return int(name[5:-4])


def i_to_name(i):
    return 'frame'+str(i)+'.png'


def p_name_to_i(name):
    return int(name[6:-7])


def p_i_to_name(i):
    return 'points'+str(i)+'.pickle'


def name_to_i_j(name):
    return int(name[8:-7].split('_')[0]), int(name[8:-7].split('_')[1])


def i_j_to_name(i, j):
    return 'matches_'+str(i)+'_'+str(j)+'.pickle'


#######################
#      Geometry       #
#######################
def ypr_to_R(ypr):
    return Rotation.from_euler('yxz',ypr, degrees=True).as_dcm()


def R_to_ypr(R):
    return Rotation.from_dcm(R).as_euler('yxz', degrees=True)


def R_t_to_RT(R, t):
    t = np.array(t).reshape((3,1))
    return np.vstack([np.hstack([R, t]), [[0, 0, 0, 1]]])


def RT_to_R_t(RT):
    R = RT[:3,:3]
    t = RT[:3,3:].flatten()
    return R, t


def RT_to_E(RT):
    R, t = RT_to_R_t(RT)
    t_cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = t_cross.dot(R)
    return E


def triangulate_points(RT, pts1, pts2):
    points = cv2.triangulatePoints(np.eye(4)[:3], RT[:3], pts1.T, pts2.T)
    points = (points / points[3]).transpose()[:, :3]
    return points


def split_R(R, n):
    stationaty_vector = np.linalg.svd(R - np.eye(3))[2][-1]
    to_stationary_x = stationaty_vector
    to_stationary_y = np.array([to_stationary_x[2], to_stationary_x[2],
                                -to_stationary_x[0] - to_stationary_x[1]])
    to_stationary_z = np.cross(to_stationary_x, to_stationary_y)
    to_stationary_x = to_stationary_x / np.linalg.norm(to_stationary_x)
    to_stationary_y = to_stationary_y / np.linalg.norm(to_stationary_y)
    to_stationary_z = to_stationary_z / np.linalg.norm(to_stationary_z)
    to_stationary_R = np.vstack(
        [to_stationary_x, to_stationary_y, to_stationary_z]).transpose()
    if np.linalg.det(to_stationary_R) < 0:
        to_stationary_R[2, :] = -to_stationary_R[2, :]
    simple_R = np.linalg.inv(to_stationary_R).dot(R).dot(to_stationary_R)
    angle = R_to_ypr(simple_R)[1]
    new_angle = angle / n
    r = to_stationary_R.dot(ypr_to_R([0, new_angle, 0])).dot(
        np.linalg.inv(to_stationary_R))
    return r


def split_RT(RT, n):
    R, T = RT_to_R_t(RT)
    r = split_R(R, n)
    polynom_r = np.zeros((3, 3))
    for i in range(n):
        polynom_r = r.dot(polynom_r) + np.eye(3)
    t = np.linalg.inv(polynom_r).dot(T)
    rt = R_t_to_RT(r, t)
    return rt


def smooth_RTs(init_RTs, n):
    RTs = []
    for RT in init_RTs:
        RTs += [split_RT(RT, n)]*n
    return RTs


def accum_RTs(RTs):
    accum_RTs = [np.eye(4)]
    for i in range(len(RTs)):
        accum_RTs.append(accum_RTs[i].dot(RTs[i]))
    return accum_RTs


def undistort_points(points):
    points = points.reshape(-1, 1, 2).astype(np.float64)
    points = cv2.undistortPoints(points, CAMERA_MATRIX, DIST_COEF)
    points = points.reshape(-1, 2)
    return points

def rectify_points(points):
    if points.shape[1] == 2:
        points = np.hstack([points, np.ones((len(points), 1))])
    return project_points(points, K=np.linalg.inv(CAMERA_MATRIX))

def unrectify_points(points):
    if points.shape[1] == 2:
        points = np.hstack([points, np.ones((len(points), 1))])
    return project_points(points)

def E_err(pts1, pts2, E):
    pts1 = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2 = np.hstack([pts2, np.ones((len(pts2), 1))])
    return np.abs(pts2.dot(E).dot(pts1.T).diagonal())

def project_points(points, K=CAMERA_MATRIX):
    proj_points = K.dot(points.T).T
    proj_points = proj_points / proj_points[:, 2].reshape(-1, 1)
    proj_points = proj_points[:, :2]
    return proj_points


#######################
#       Files         #
#######################
def read_image(name):
    im = plt.imread(name, 0)
    im = np.flip(im, axis=0)
    return im


def read_frame(i):
    return read_image(os.path.join(ROOT_PATH, FRAMES_PATH, i_to_name(i)))


def read_points(i):
    with open(os.path.join(ROOT_PATH, POINTS_PATH, p_i_to_name(i)),mode='rb') as f:
        d = pickle.load(f)
    points = d['points']
    descs = d['descs']
    point_colors = d['point_colors']
    return points, descs, point_colors


def write_points(i, points, descs, point_colors):
    with open(os.path.join(ROOT_PATH, POINTS_PATH, p_i_to_name(i)), mode='wb') as f:
        pickle.dump({'points': points, 'descs': descs, 'point_colors': point_colors}, f)


def read_matches(i, j):
    with open(os.path.join(ROOT_PATH, MATCHES_PATH, i_j_to_name(i, j)), mode='rb') as f:
        d = pickle.load(f)
    RT = d['RT']
    matches = d['matches']
    pts_coord = d['pts_coord']
    return RT, matches, pts_coord


def write_matches(i, j, RT, matches, pts_coord):
    with open(os.path.join(ROOT_PATH, MATCHES_PATH, i_j_to_name(i, j)), mode='wb') as f:
        pickle.dump({'RT': RT, 'matches': matches, 'pts_coord': pts_coord}, f)