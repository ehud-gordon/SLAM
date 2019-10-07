from itertools import cycle
from PARAMS import *
from utils import *
from matplotlib.colors import cnames
from create_rts import *
from match_frames import initial_matches, findE, findRT
colors = cycle(list(cnames.keys()))


# PARAMS
n1 = 0
n2 = 10


class MouseMonitor:
    def __init__(self, n1, n2):
        self.im1 = read_frame(n1)
        self.im2 = read_frame(n2)
        # self.RT, self.matches, self.pts_coord = read_matches(n1, n2)
        self.points1, self.descs1, _ = read_points(n1)
        self.points2, self.descs2, _ = read_points(n2)
        # self.E = RT_to_E(self.RT)
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.ax1.imshow(self.im1, origin='lower')
        self.ax1.set_xlim([0.0, self.im1.shape[1]])
        self.ax1.set_ylim([0.0, self.im1.shape[0]])
        self.ax1.autoscale = False
        self.ax2.imshow(self.im2, origin='lower')
        self.ax2.set_xlim([0.0, self.im2.shape[1]])
        self.ax2.set_ylim([0.0, self.im2.shape[0]])
        self.ax2.autoscale = False
        self.plots1 = []
        self.plots2 = []
        self.stage = 0
        self.pts1 = None
        self.pts2 = None
        self.calc_E = None
        self.epipolars = []

    def clear1(self):
        for p in self.plots1:
            p.remove()
        self.plots1 = []

    def clear2(self):
        for p in self.plots2:
            p.remove()
        self.plots2 = []
    def clear_epipolars(self):
        for p in self.epipolars:
            p.remove()
        self.epipolars = []

    def plot_matches(self, pts1, pts2):
        for (x1, y1), (x2, y2), c in zip(pts1, pts2, colors):
            self.plots1.append(self.ax1.scatter(x1, y1, c=c, marker='.', s=64))
            self.plots2.append(self.ax2.scatter(x2, y2, c=c, marker='.', s=64))

    def __call__(self, event):
        x = event.xdata
        y = event.ydata
        if event.inaxes is None:
            idx = -1
        elif event.inaxes.axes == self.ax1:
            idx = 0
        else:
            idx = 1
        # print('idx=', idx)
        SHOW_ORIGINAL_POINTS_KEY = 'q'
        SHOW_UNDISTORTED_POINTS_KEY = 'w'
        INITIAL_MATCH_KEY = 'a'
        ESSENTIAL_MATCH_KEY = 'z'
        RT_MATCH_KEY = 'x'
        CLEAR_KEY = 'c'
        if event.name == 'button_press_event' and idx != -1 and event.button == 1 and self.stage >= 4:
            F = np.linalg.inv(CAMERA_MATRIX).T.dot(self.calc_E).dot(np.linalg.inv(CAMERA_MATRIX))
            # E_err(unrectify_points(self.pts_ransac1), unrectify_points(self.pts_ransac2), F)
            a, b, c = cv2.computeCorrespondEpilines(np.array([[x, y]]).reshape(-1, 1, 2), idx + 1, F)[0].flatten()
            # y = mx + d
            d = -c / b
            m = -a / b
            line_xs = np.array([0, self.im1.shape[1] - 1])
            line_ys = m*line_xs + d
            # print(line_xs, line_ys)
            # print('y = {}*x + {}'.format(m,d))
            if idx == 0:
                self.epipolars.append(self.ax2.plot(line_xs, line_ys)[0])
            elif idx == 1:
                self.epipolars.append(self.ax1.plot(line_xs, line_ys)[0])

        elif event.name == 'key_press_event':
            print(event.key)
            if event.key == SHOW_ORIGINAL_POINTS_KEY:
                self.pts1 = self.plots1
                self.pts2 = self.plots2
                self.stage = 1
                self.plots1.append(self.ax1.scatter(self.points1[:, 0], self.points1[:, 1], c='r', marker='.', s=4))
                self.plots2.append(self.ax2.scatter(self.points2[:, 0], self.points2[:, 1], c='r', marker='.', s=4))
                # self.ax2.plot([0, 100], [0, 1000])
            if event.key == SHOW_UNDISTORTED_POINTS_KEY:
                self.undistorted_points1 = rectify_points(self.points1)
                self.undistorted_points2 = rectify_points(self.points2)
                self.stage = 2
                ps1 = unrectify_points(self.undistorted_points1)
                ps2 = unrectify_points(self.undistorted_points2)
                self.plots1.append(self.ax1.scatter(ps1[:, 0], ps1[:, 1], c='r', marker='.', s=4))
                self.plots2.append(self.ax2.scatter(ps2[:, 0], ps2[:, 1], c='r', marker='.', s=4))
            if event.key == INITIAL_MATCH_KEY:
                self.pts_initial1, self.pts_initial2, self.matches_initial = initial_matches(self.undistorted_points1, self.undistorted_points2, self.descs1, self.descs2)
                print('done')
                self.plot_matches(self.pts_initial1, self.pts_initial2)
                self.stage = 3
                ps1 = unrectify_points(self.pts_initial1)
                ps2 = unrectify_points(self.pts_initial2)
                self.clear1()
                self.clear2()
                self.plot_matches(ps1, ps2)
            if event.key == ESSENTIAL_MATCH_KEY:
                self.calc_E, self.pts_ransac1, self.pts_ransac2, self.matches_ransac = findE(self.pts_initial1, self.pts_initial2, self.matches_initial)
                print(np.array(sorted(self.pts_ransac1, key=lambda q: q[0])))
                print('done')
                self.stage = 4
                self.clear1()
                self.clear2()
                ps1 = unrectify_points(self.pts_ransac1)
                ps2 = unrectify_points(self.pts_ransac2)
                self.plot_matches(ps1, ps2)
            if event.key == RT_MATCH_KEY:
                self.calc_RT, self.pts_pose1, self.pts_pose2, self.matches_pose = findRT(self.calc_E, self.pts_ransac1, self.pts_ransac2, self.matches_ransac)
                R, t = RT_to_R_t(self.calc_RT)
                print('ypr',R_to_ypr(R))
                print('t',t)
                print('done')
                self.stage = 5
                self.clear1()
                self.clear2()
                ps1 = unrectify_points(self.pts_pose1)
                ps2 = unrectify_points(self.pts_pose2)
                self.plot_matches(ps1, ps2)
            if event.key == 'v':
                pts_3d = triangulate_points(self.calc_RT, self.pts_pose1, self.pts_pose2)
                cccc = len(pts_3d)
                pts_3d = np.array(sorted(pts_3d, key=lambda aaaa: aaaa[2]))
                pts_3d_proj1 = project_points(pts_3d)
                aaa=np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
                bbb=self.calc_RT.dot(aaa.T).T
                bbb=bbb[:, :3]
                pts_3d_proj2 = project_points(bbb)
                self.ax1.scatter(pts_3d_proj1[:cccc//2, 0], pts_3d_proj1[:cccc//2, 1], c='r',marker='x', s=32)
                self.ax2.scatter(pts_3d_proj2[:cccc//2, 0], pts_3d_proj2[:cccc//2, 1], c='r',marker='x', s=32)
                self.ax1.scatter(pts_3d_proj1[cccc // 2:, 0], pts_3d_proj1[cccc // 2:, 1], c='b', marker='x', s=32)
                self.ax2.scatter(pts_3d_proj2[cccc // 2:, 0], pts_3d_proj2[cccc // 2:, 1], c='b', marker='x', s=32)
            # if event.key == 'q':
            #     pts1, pts2, _ = initial_matches(self.points1, self.points2, self.descs1, self.descs2)
            #     self.plot_matches(pts1, pts2)
            #
            if event.key == CLEAR_KEY:
                self.clear_epipolars()
        self.fig.canvas.draw_idle()
    # def find_closest_point(self, x, y, idx):
    #     if self.points[idx]:
    #         xs = np.array([p[0] for p in self.points[idx]])
    #         ys = np.array([p[1] for p in self.points[idx]])
    #         dists = ((xs - x)**2 + (ys - y)**2)**0.5
    #         return np.argmin(dists)

def main():
    mouse = MouseMonitor(n1, n2)
    mouse_cid = mouse.fig.canvas.mpl_connect('button_press_event', mouse)
    keyboard_cid = mouse.fig.canvas.mpl_connect('key_press_event', mouse)
    plt.show()


if __name__ == '__main__':
    main()
