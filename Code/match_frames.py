from PARAMS import *
from utils import *


def initial_matches(pts1, pts2, descs1, descs2):
    print('before any filter =', len(pts1), len(pts2))
    bf = cv2.BFMatcher()
    matches_knn = bf.knnMatch(descs1, descs2, k=2)
    matches = []
    for m, n in matches_knn:
        if m.distance < SIFT_DESC_RATIO * n.distance:
            matches.append([m.queryIdx, m.trainIdx])
    matches = np.array(matches, dtype=np.int)
    return pts1[matches[:, 0]], pts2[matches[:, 1]], matches


def findE(pts1, pts2, matches):
    x = len(pts1)
    print('before RANSAC =',len(pts1))
    E, mask = cv2.findEssentialMat(pts1, pts2, np.eye(3), cv2.RANSAC, prob=0.999, threshold=RANSAC_THRESHOLD)
    mask = mask.flatten().astype(np.bool)
    print('after RANSAC =', mask.sum())
    # if mask.sum() != x:
    #     1/0
    return E, pts1[mask], pts2[mask], matches[mask]


def findRT(E, pts1, pts2, matches):
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, np.eye(3))
    mask = (mask.flatten() // 255).astype(np.bool)
    print('mask=', mask.sum())
    # mask[:] = True
    RT = R_t_to_RT(R, t)
    return RT, pts1[mask], pts2[mask], matches[mask]


def match_images(sift_points1, sift_points2, descs1, descs2):
    pts1 = rectify_points(sift_points1)
    pts2 = rectify_points(sift_points2)
    pts1, pts2, matches = initial_matches(pts1, pts2, descs1, descs2)
    E, pts1, pts2, matches = findE(pts1, pts2, matches)
    RT, pts1, pts2, matches = findRT(E, pts1, pts2, matches)
    pts_3d = triangulate_points(RT, pts1, pts2)
    pts_coord = np.hstack([pts1, pts2, pts_3d])
    return RT, matches, pts_coord


def main():
    num_of_frames = len([0 for name in os.listdir(os.path.join(ROOT_PATH,FRAMES_PATH)) if is_frame_name(name)])
    kf_names = [i_to_name(i) for i in range(START_FRAME, num_of_frames, FRAMES_STEP)]
    times_lengths = []
    end_time = time.time()
    for i in range(len(kf_names) - NUM_OF_KF_TO_FOLLOW + 1):
        start_time = end_time
        idx1 = name_to_i(kf_names[i])
        points1, descs1, _ = read_points(idx1)
        for j in range(i + 1, i + NUM_OF_KF_TO_FOLLOW):
            idx2 = name_to_i(kf_names[j])
            print(idx1, idx2)
            points2, descs2, _ = read_points(idx2)
            matches_name = i_j_to_name(idx1, idx2)
            RT, matches, pts_coord = match_images(points1, points2, descs1, descs2)
            R, t = RT_to_R_t(RT)
            roll, pitch, yaw = R_to_ypr(R)
            if VERBOSE:
                print('--- found {} matches from {} to {}'.format(len(matches),kf_names[i],kf_names[j]))
                print('--- yaw={}, pitch={}, roll={}'.format(yaw, pitch, roll))
                print('--- x={}, y={}, z={}'.format(t[0], t[1], t[2]))
                print('--- saving as {}'.format(matches_name))
            write_matches(idx1, idx2, RT, matches, pts_coord)
        end_time = time.time()
        times_lengths.append(end_time - start_time)
        time_examples = times_lengths[:min(len(times_lengths), EXAMPLES_FOR_ETA)]
        avg_time = sum(time_examples) / len(time_examples)
        frames_left = (len(kf_names) - NUM_OF_KF_TO_FOLLOW - i)
        eta = frames_left * avg_time
        print('done frame {}/{}'.format(i+1, len(kf_names) - NUM_OF_KF_TO_FOLLOW + 1))
        print('completed {}%'.format(round(100 * (i+1) / (len(kf_names) - NUM_OF_KF_TO_FOLLOW + 1), 2)))
        print('ETA {} seconds'.format(round(eta, 1)))
        print()


if __name__ == '__main__':
    main()
