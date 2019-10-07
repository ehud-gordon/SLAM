from PARAMS import *
from utils import *
from scipy.sparse import lil_matrix


def get_RTs(diff_size):
    assert NUM_OF_KF_TO_FOLLOW > diff_size
    diff_func = lambda x: x[1] - x[0]
    matches_names = [matches_name for matches_name in os.listdir(os.path.join(ROOT_PATH, MATCHES_PATH)) if is_match_name(matches_name) and diff_func(name_to_i_j(matches_name)) == diff_size*FRAMES_STEP]
    num_of_kf_pairs = len(set([matches_name.split('_')[1] for matches_name in matches_names]))
    RTs = []
    for i in range(num_of_kf_pairs):
        RTs.append(read_matches(i * FRAMES_STEP, (i + diff_size) * FRAMES_STEP)[0])
    return RTs


def get_point_matches():
    matches_names = [matches_name for matches_name in os.listdir(os.path.join(ROOT_PATH, MATCHES_PATH)) if is_match_name(matches_name)]
    num_of_kf_pairs = len(set([matches_name.split('_')[1] for matches_name in matches_names]))
    total_matches = {}
    all_RTs = [np.eye(4)]
    curr_RT = np.eye(4)
    for frame_idx in range(num_of_kf_pairs):
        frame = frame_idx * FRAMES_STEP
        _, _, point_colors = read_points(frame)
        RT, matches, pts_coord = read_matches(frame, frame + FRAMES_STEP)
        for (i, j), curr_pts_coord, color in zip(matches, pts_coord, point_colors):
            curr_accur = (frame, i)
            next_accur = (frame + FRAMES_STEP, j)
            pts1 = curr_pts_coord[:2]
            pts2 = curr_pts_coord[2:4]
            curr_pts_3d = curr_pts_coord[4:]
            curr_pts_3d = np.linalg.inv(curr_RT).dot(np.append(curr_pts_3d,1))[:3]
            if curr_accur in total_matches:
                start_frame, color_avg, pts_3d, all_accur = total_matches.pop(curr_accur)
                pts_3d = ((len(all_accur)-1)*pts_3d + curr_pts_3d) / len(all_accur)
                # color_avg = ((len(all_accur)-1)*color_avg + color) / len(all_accur)
                all_accur.append((j, pts2))
                total_matches[next_accur] = start_frame, color_avg, pts_3d, all_accur
            else:
                total_matches[next_accur] = frame, color, curr_pts_3d, [(i, pts1),(j, pts2)]
        curr_RT = RT.dot(curr_RT)
        all_RTs.append(curr_RT)
    total_matches = list(filter(lambda x: len(x[3])>2, total_matches.values()))
    return all_RTs, total_matches

def calc_res(all_params, num_of_cameras, num_of_points, total_matches):
    camera_params = all_params[:num_of_cameras * 6].reshape((num_of_cameras, 6))
    points_3d = all_params[num_of_cameras * 6:].reshape((num_of_points, 3))
    reses = []
    for point_idx, (start_frame, color, pts_3d, accurs) in enumerate(total_matches):
        pts_3d = points_3d[point_idx]
        for accur_idx, accur in enumerate(accurs):
            camera_param = camera_params[start_frame//FRAMES_STEP + accur_idx]
            RT = R_t_to_RT(ypr_to_R(camera_param[:3]), camera_param[3:])
            pts_3d = np.linalg.inv(RT).dot(np.append(pts_3d,1))[:3]
            res = (pts_3d[:2] / pts_3d[2]) - accur[1]
            reses.append(res)
    return np.array(reses).flatten()


def foo():
    all_RTs, total_matches = get_point_matches()
    all_yprs = np.array([R_to_ypr(RT_to_R_t(RT)[0]) for RT in all_RTs])
    all_ts = np.array([RT_to_R_t(RT)[1] for RT in all_RTs])
    all_camera_params = np.hstack([all_yprs, all_ts])
    all_3d_points = np.array([p[2] for p in total_matches])
    all_params = np.append(all_camera_params.flatten(),all_3d_points.flatten())
    num_of_points = len(total_matches)
    num_of_obs = sum([len(p[3]) for p in total_matches])
    num_of_res = num_of_obs * 2
    num_of_cameras = len(all_RTs)
    num_of_params = num_of_cameras*6 + num_of_points*3
    reses = calc_res(all_params, num_of_cameras, num_of_points, total_matches)
    # plt.plot(reses)
    # plt.show()
    A = lil_matrix((num_of_res, num_of_params), dtype=int)
    obs_idx = 0
    for point_idx, (start_frame, color, pts_3d, accurs) in enumerate(total_matches):
        for i in range(len(accurs)):
            j = i + start_frame//FRAMES_STEP
            for k in range(6):
                A[2 * obs_idx, j*6 + k] = 1
                A[2 * obs_idx + 1, j*6 + k] = 1
            for k in range(3):
                A[2 * obs_idx, num_of_cameras*6 + point_idx*3 + k] = 1
                A[2 * obs_idx + 1, num_of_cameras*6 + point_idx*3 + k] = 1
            obs_idx += 1
    from scipy.optimize import least_squares
    import time
    t0 = time.time()
    print('starting')
    result = least_squares(calc_res, all_params,jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', max_nfev=10, args=(num_of_cameras, num_of_points, total_matches))
    final_params = result.x
    # final_params = all_params
    print('took', time.time()-t0)
    # plt.plot(result.fun)
    # plt.show()
    camera_params = final_params[:num_of_cameras * 6].reshape((num_of_cameras, 6))
    points_3d = final_params[num_of_cameras * 6:].reshape((num_of_points, 3))
    RTs = []
    for i in range(len(camera_params)):
        R = ypr_to_R(camera_params[i][:3])
        t = camera_params[i][3:]
        RT = R_t_to_RT(R, t)
        RTs.append(RT)
    return RTs, points_3d
# def get_points():
#     init_RTs = get_RTs(1)
#     RTs = accum_RTs(init_RTs)
#     points_matches =
#     for p in

def no_translation(RTs):
    new_RTs = []
    for RT in RTs:
        R, t = RT_to_R_t(RT)
        RT = R_t_to_RT(R, [0, 0, 0])
        new_RTs.append(RT)
    return new_RTs


def no_rotation(RTs):
    new_RTs = []
    for RT in RTs:
        R, t = RT_to_R_t(RT)
        RT = R_t_to_RT(np.eye(3), t)
        new_RTs.append(RT)
    return new_RTs


def create_mean_rts():
    RTs_12 = get_RTs(1)
    RTs_13 = get_RTs(2)
    RTs_14 = get_RTs(3)
    new_RTs = []
    for i in range(len(RTs_12)-2):
        RT_12 = RTs_12[i]
        RT_13_32 = RTs_13[i].dot(np.linalg.inv(RTs_12[i+1]))
        RT_14_42 = RTs_14[i].dot(np.linalg.inv(RTs_13[i+1]))
        ypr_12 = R_to_ypr(RT_to_R_t(RT_12)[0])
        ypr_13_32 = R_to_ypr(RT_to_R_t(RT_13_32)[0])
        ypr_14_42 = R_to_ypr(RT_to_R_t(RT_14_42)[0])
        yprs = np.vstack([ypr_12, ypr_13_32, ypr_14_42])
        print(yprs)
        new_ypr = np.mean(yprs, axis=0)
        new_R = ypr_to_R(new_ypr)
        new_RT = R_t_to_RT(new_R, RT_to_R_t(RT_12)[1])
        new_RTs.append(new_RT)
    return new_RTs

# def create_mean_rts2():
#     RTs_lists  = [get_RTs(i+1) for i in range(NUM_OF_KF_TO_FOLLOW-1)]
#     new_RTs = []
#     for i in range(len(RTs_lists[-1])):
#         RTs12 = [RTs_lists[0]]
#         for j in range(len(RTs_lists)-1):
#             RTs12.append(RTs_lists[j+1].dot(np.linalg.inv(RTs_lists[j+1]
#                                                           )))
#         RT_12 = RTs_12[i]
#         RT_13_32 = RTs_13[i].dot(np.linalg.inv(RTs_12[i+1]))
#         RT_14_42 = RTs_14[i].dot(np.linalg.inv(RTs_13[i+1]))
#         ypr_12 = R_to_ypr(RT_to_R_t(RT_12)[0])
#         ypr_13_32 = R_to_ypr(RT_to_R_t(RT_13_32)[0])
#         ypr_14_42 = R_to_ypr(RT_to_R_t(RT_14_42)[0])
#         yprs = np.vstack([ypr_12, ypr_13_32, ypr_14_42])
#         print(yprs)
#         new_ypr = np.mean(yprs, axis=0)
#         new_R = ypr_to_R(new_ypr)
#         new_RT = R_t_to_RT(new_R, RT_to_R_t(RT_12)[1])
#         new_RTs.append(new_RT)
#     return new_RTs


# def retriangulate_points():




# def find_t_scales():
#     scales = [1]
#     for i in range(num_of_kf_pairs - 1):
#         R12 = Rs_12[i]
#         R23 = Rs_12[i+1]
#         R13 = Rs_13[i]
#         print('R23 * R12=', R23.dot(R12))
#         print('R13=', R13)
#         t12 = ts_12[i]
#         t23 = ts_12[i+1]
#         t13 = ts_13[i]
#         u,s,v_t = np.linalg.svd(np.hstack([R23.dot(t12), t23, -t13]))
#         a, b, c = v_t[-1]
#         curr_scale = scales[i]
#         next_scale = curr_scale*b/a
#         scales.append(next_scale)
#     RTs12 = []
#     for i in range(num_of_kf_pairs):
#         R12 = Rs_12[i]
#         t12 = ts_12[i]
#         s = scales[i]
#         RT12 = R_t_to_RT(R12, s*t12)
#         RTs12.append(RT12)
#     return RTs12
#     with open(os.path.join(ROOT_PATH, RTS_NAME), mode='wb') as f:
#         pickle.dump(RTs_to_start_frame, f)


def create_zigzag_rts():
    return [
        R_t_to_RT(ypr_to_R([-10, 0, 0]), [0, 0, 0]),
        R_t_to_RT(ypr_to_R([10, 0, 0]), [0, 0, 1]),
        R_t_to_RT(ypr_to_R([10, 0, 0]), [0, 0, 1]),
        R_t_to_RT(ypr_to_R([-10, 0, 0]), [0, 0, 1]),
        R_t_to_RT(ypr_to_R([-10, 0, 0]), [0, 0, 1]),
        R_t_to_RT(ypr_to_R([10, 0, 0]), [0, 0, 1]),
        R_t_to_RT(ypr_to_R([0, 15, 0]), [0, 0, 1]),
        R_t_to_RT(ypr_to_R([0, -15, 0]), [0, 0, 1]),
        R_t_to_RT(ypr_to_R([0, -15, 0]), [0, 0, 1]),
        R_t_to_RT(ypr_to_R([0, 15, 0]), [0, 0, 1]),
        R_t_to_RT(ypr_to_R([0, 15, 0]), [0, 0, 1]),
        R_t_to_RT(ypr_to_R([0, -15, 0]), [0, 0, 1]),
    ]


def main():
    n = 10
    R = ypr_to_R([10,30,40])
    T = [12,34,56]
    RT = R_t_to_RT(R,T)
    rt = split_RT(RT, n)
    new_RT = np.linalg.multi_dot([rt] * n)
    print('RT=\n',RT)
    print('rt=\n', rt)
    print('new_RT=\n', new_RT)
    r, t = RT_to_R_t(rt)
    new_R, new_T = RT_to_R_t(new_RT)
    print('YPR=\n', R_to_ypr(R))
    print('ypr=\n', R_to_ypr(r))
    print('new_YPR=\n', R_to_ypr(new_R))
    print('T=\n', T)
    print('t=\n', t)
    print('new_T=\n', new_T)
if __name__ == '__main__':
    # main()
    foo()