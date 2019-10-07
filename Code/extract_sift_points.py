from PARAMS import *
from utils import *


def extract_points(image):
    sift = cv2.xfeatures2d.SIFT_create(NUM_OF_SIFT_POINTS)
    sift_points, sift_descs = sift.detectAndCompute(image, None)
    sift_points = np.array([p.pt for p in sift_points])
    rounded_points = np.round(sift_points).astype(np.int)
    point_colors = image[rounded_points[:,1],rounded_points[:,0]]
    return sift_points, sift_descs, point_colors


def display_sift_points(idx):
    im = read_frame(idx)
    # points = read_points(idx)[0]
    points = extract_points(im)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im, origin='lower')
    ax.set_xlim([0.0, im.shape[1]])
    ax.set_ylim([0.0, im.shape[0]])
    ax.autoscale = False
    plt.scatter(points[:,0], points[:,1], c='r', marker='.')
    plt.show()


def main():
    num_of_frames = len([0 for name in os.listdir(os.path.join(ROOT_PATH, FRAMES_PATH)) if is_frame_name(name)])
    kf_names = [i_to_name(i) for i in range(START_FRAME, num_of_frames, FRAMES_STEP)]
    times_lengths = []
    end_time = time.time()
    for i in range(len(kf_names)):
        start_time = end_time
        idx = name_to_i(kf_names[i])
        im = read_frame(idx)
        points_name = p_i_to_name(idx)
        points, descs, point_colors = extract_points(im)
        if VERBOSE:
            print('--- found {} points in frame {}'.format(len(points), idx)),
            print('--- saving as {}'.format(points_name))
        write_points(idx, points, descs, point_colors)
        end_time = time.time()
        times_lengths.append(end_time - start_time)
        time_examples = times_lengths[
                        :min(len(times_lengths), EXAMPLES_FOR_ETA)]
        avg_time = sum(time_examples) / len(time_examples)
        frames_left = (len(kf_names) - i - 1)
        eta = frames_left * avg_time
        print('done frame {}/{}'.format(i + 1, len(kf_names)))
        print('completed {}%'.format(round(100 * (i + 1) / len(kf_names),2)))
        print('ETA {} seconds'.format(round(eta, 1)))
        print()


if __name__ == '__main__':
    main()
    # display_sift_points(0)