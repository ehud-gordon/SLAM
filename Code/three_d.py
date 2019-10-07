#!C:\Users\idob\AppData\Local\Programs\Python\Python35-32\python
import numpy as np
import pygame, math, sys, random, time, winsound
from pygame.locals import *
import random as r
from copy import deepcopy

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
SCALING = 1
WIDTH = 448
HIGHT = 470
DEPTH = 625
EDTH = 625
SPEED = 100
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
NUM_OF_DOTS = 1000
NUM_OF_LINES = 0
NUM_OF_CUBES = 30
NUM_OF_PYRAMIDS = 0


def draw_dots(dots):
    projected_dots = np.zeros((dots.shape[0],4))
    zs = np.zeros((dots.shape[0],2))
    zs[:,0] = dots[:, 2]
    zs[:, 1] = dots[:, 2]
    projected_dots[:,2] = dots[:, 2]
    projected_dots[:, 3] = np.int32(np.round(255 / (0.005 * dots[:,2] + 1)))
    projected_dots[:,:2] = np.int32(np.round(dots[:, :2]/(0.0016*zs) - (-WIDTH // 2)))
    for x, y, z, c in projected_dots:
        s = 1
        if c > 230:
            s = 4
        elif c > 180:
            s = 3
        elif c > 100:
            s = 2
        if z > 0 and x >= 0 and x < WIDTH and y >= 0 and y < HIGHT:
            screen.fill((c,c,c), rect=pygame.rect.Rect((x, y), (s, s)))

def draw_dots_with_color(dots, dot_colors):
    projected_dots = np.zeros((dots.shape[0],4))
    zs = np.zeros((dots.shape[0],2))
    zs[:,0] = dots[:, 2]
    zs[:, 1] = dots[:, 2]
    projected_dots[:,2] = dots[:, 2]
    projected_dots[:, 3] = np.int32(np.round(255 / (0.005 * dots[:,2] + 1)))
    projected_dots[:,:2] = np.int32(np.round(dots[:, :2]/(0.0016*zs) - (-WIDTH // 2)))
    for (x, y, z, c), clr in zip(projected_dots, dot_colors):
        s = 1
        if c > 230:
            s = 4
        elif c > 180:
            s = 3
        elif c > 100:
            s = 2
        if z > 0 and x >= 0 and x < WIDTH and y >= 0 and y < HIGHT:
            screen.fill(clr, rect=pygame.rect.Rect((x, y), (s, s)))


def draw_dots_with_color2(dots, dot_colors, K):
    projected_dots = project_points(dots, K)
    brightness = np.int32(np.round(255 / (0.005 * dots[:, 2] + 1)))
    displayed_dots = np.int32(np.round(projected_dots))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xlim([0.0, WIDTH])
    # ax.set_ylim([0.0, HIGHT])
    # ax.autoscale = False
    if dot_colors is None:
        dot_colors = brightness.reshape(-1, 1)
        dot_colors = np.hstack([dot_colors, dot_colors, dot_colors])
    for (x, y), z, clr in zip(displayed_dots,dots[:,2], dot_colors):
        if z > 0 and x >= 0 and x < WIDTH and y >= 0 and y < HIGHT:
            screen.fill(clr, rect=pygame.rect.Rect((SCALING*x, SCALING*(HIGHT - 1 - y) ), (2, 2)))
    #         ax.plot(x, y, color='r', marker='.')
    # fig.show()
def draw_lines2(lines, K, is_blue=False):
    projected_lines = project_points(lines, K)
    displayed_lines = np.int32(np.round(projected_lines))
    for i in range(len(lines)//2):
        p1 = displayed_lines[2 * i]
        p2 = displayed_lines[2 * i + 1]
        z1 = lines[2 * i][2]
        z2 = lines[2 * i + 1][2]
        if p1[0] >= 0 and p1[1] >= 0 and p1[0] < (WIDTH-1) and p1[1] < (HIGHT-1):
            if p2[0] >= 0 and p2[1] >= 0 and p2[0] < (WIDTH - 1) and p2[1] < (HIGHT - 1):
                if z1 > 0 and z2 > 0:
                    x1 = SCALING*p1[0]
                    y1 = SCALING*(HIGHT-1-p1[1])
                    x2 = SCALING * p2[0]
                    y2 = SCALING * (HIGHT - 1 - p2[1])
                    pygame.draw.line(screen, RED if not is_blue else BLUE, (x1,y1), (x2,y2), 1)

def make_tranformation(trans, x, y, z, a):
    ANGLE = 1 * a
    translation = np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])
    rotaton = np.array(
        [[np.cos(ANGLE * np.pi / 180), 0, np.sin(ANGLE * np.pi / 180), 0],
         [0, 1, 0, 0], [np.cos((ANGLE + 90) * np.pi / 180), 0,
                     np.sin((ANGLE + 90) * np.pi / 180), 0], [0,0,0,1]])
    return np.dot(rotaton, np.dot(translation, trans))

def transform(dots, trans):
    dots_copy = np.zeros((dots.shape[0], 4),dtype=np.float64)
    dots_copy[:, :3] = dots
    dots_copy[:, 3] = 1
    dots_copy = np.dot(trans, dots_copy.transpose()).transpose()[:,:3]
    return dots_copy
    return np.int32(np.round(dots_copy))



def draw_lines(lines):
    projected_lines = np.zeros((lines.shape[0], 4))
    zs = np.zeros((lines.shape[0], 2))
    zs[:, 0] = lines[:, 2]
    zs[:, 1] = lines[:, 2]
    projected_lines[:, 2] = lines[:, 2]
    projected_lines[:, 3] = np.int32(np.round(255 / (0.005 * lines[:, 2] + 1)))
    projected_lines[:, :2] = np.int32(
        np.round(lines[:, :2] / (0.0016 * zs) - (-WIDTH // 2)))
    for i in range(len(projected_lines)//2):
        x1 = projected_lines[2*i,0]
        y1 = projected_lines[2 * i, 1]
        z1 = projected_lines[2 * i, 2]
        x2 = projected_lines[2 * i+1, 0]
        y2 = projected_lines[2 * i+1, 1]
        z2 = projected_lines[2 * i+1, 2]
        if z1 > 0 and z2 > 0:
            pygame.draw.line(screen, RED, (x1, y1), (x2, y2), 1)


pygame.init()
screen = pygame.display.set_mode((SCALING*WIDTH, SCALING*HIGHT), DOUBLEBUF)
clock = pygame.time.Clock()
dots = np.array([[r.randint(-WIDTH // 2, WIDTH // 2 - 1),
                  r.randint(-HIGHT // 2, HIGHT // 2 - 1),
                  r.randint(-DEPTH // 2, DEPTH - 1)] for i in
                 range(NUM_OF_DOTS)])
lines = [[r.randint(-WIDTH // 2, WIDTH // 2 - 1),
          r.randint(-HIGHT // 2, HIGHT // 2 - 1), r.randint(0, DEPTH - 1)] for
         i in range(2 * NUM_OF_LINES)]
# cube_pos = [[r.randint(10, 100), r.randint(-WIDTH // 2, WIDTH // 2 - 1),
#           r.randint(-HIGHT // 2, HIGHT // 2 - 1), r.randint(0, DEPTH - 1), r.randint(-90, 90)] for
#          i in range(NUM_OF_CUBES)]
cube_pos = [[r.choice([0.5,0.7,1]), 10*r.random(),
             10 *r.random(), 10*r.random(), r.randint(-90, 90)] for
         i in range(NUM_OF_CUBES)]
# pyramid_pos = [[r.randint(10, 100), r.randint(-WIDTH // 2, WIDTH // 2 - 1),
#           r.randint(-HIGHT // 2, HIGHT // 2 - 1), r.randint(0, DEPTH - 1), 0] for
#          i in range(NUM_OF_PYRAMIDS)]
pyramid_pos = [[r.choice([0.5,0.7,1]), 10*r.random(),
             10 *r.random(), 10*r.random(), r.randint(-90, 90)] for
         i in range(NUM_OF_PYRAMIDS)]

def make_cube(s):
    cube = []
    DIMS = 3
    for i in range(2 ** DIMS):
        for j in range(2 ** DIMS):
            a = bin(i)[2:].zfill(DIMS)
            b = bin(j)[2:].zfill(DIMS)
            diffs = 0
            for k in range(DIMS):
                if a[k] != b[k]:
                    diffs += 1
            if diffs == 1:
                cube.append([int(a[0]) * s, int(a[1]) * s,
                             int(a[2]) * s])
                cube.append([int(b[0]) * s, int(b[1]) * s,
                             int(b[2]) * s])
    return cube
def make_pyramid(s):
    return [[0,0,0],[s,0,0],[0,0,0],[0,0,s],[s,0,0],[s,0,s],[0,0,s],[s,0,s],[0,0,0],[s/2,-s,s/2],[s,0,0],[s/2,-s,s/2],[0,0,s],[s/2,-s,s/2],[s,0,s],[s/2,-s,s/2]]

for s,x,y,z,a in cube_pos:
    cube = list(transform(np.array(make_cube(s)),
                           make_tranformation(np.eye(4), x, y, z, a)))
    lines += cube
for s,x,y,z,a in pyramid_pos:
    pyramid = list(transform(np.array(make_pyramid(s)),
                           make_tranformation(np.eye(4), x, y, z, a)))
    lines += pyramid

lines = np.array(lines)
direction = None
direction2 = None
move = [0, 0, 0]
move2 = np.array([0, 0, 0, 0])
v = 0
y = 0
x = 0
z = 0
vz = 0
vx = 0
a = 0
va = 0
trans = np.eye(4)
#######
from create_rts import *
from utils import *
import match_frames_old
frame_idx = 0
smooth_factor = 10
init_RTs = get_RTs(1)
# init_RTs = create_zigzag_rts()
# init_RTs = create_mean_rts()
RTs = accum_RTs(init_RTs)
RTs = accum_RTs(smooth_RTs(init_RTs, smooth_factor))
# RTs = no_translation(RTs)
# RTs = no_rotation(RTs)
# RTs = create_basic_rts()

# dots = []
# dot_colors = []
# last_RT = np.eye(4)
# for i, RT in enumerate(RTs):
#     if i <= 0:
#         n1 = i * FRAMES_STEP
#         n2 = (i+1) * FRAMES_STEP
#         _, matches, pts_coord = read_matches(n1, n2)
#         dots.append(np.linalg.inv(last_RT).dot(np.vstack([pts_coord[:, -3:].T, np.ones((1,len(pts_coord)))])).T[:,:3])
#         dot_colors.append(read_points(n1)[2][matches[:, 0]])
#         last_RT = RT
# dots = np.vstack(dots)
# dot_colors = np.vstack(dot_colors)
if False:
    RTs, dots = foo()
    dot_colors = np.array([WHITE] * len(dots))
else:
    if True:
        all_RTs, total_matches = get_point_matches()
        dots = np.array([p[2] for p in total_matches])
        dot_colors = np.array([p[1] for p in total_matches])
        # dot_colors = np.array([WHITE]*len(dots))
    else:
        n1 = 0*FRAMES_STEP
        n2 = 1*FRAMES_STEP
        RT, matches, pts_coord = read_matches(n1, n2)
        dots = pts_coord[:, -3:]
        _, _, dot_colors = read_points(n1)
        dot_colors = dot_colors[matches[:, 0]]
assert dots.shape[0] == dot_colors.shape[0]
# dots[:,2] = 200*dots[:,2]
K = CAMERA_MATRIX.copy()
print(len(RTs))
print(dots.shape)
camera_cube = np.array(make_cube(0.5))
# camera_cube = transform(np.array(make_cube(0.5)), make_tranformation(np.eye(4), 0, 0, 5, 0))
print(camera_cube)
#######
mode = '1st person'
while True:
    screen.fill(BLACK)
    got_input = False
    direction = None
    direction2 = None
    # while not got_input:
    clock.tick(SPEED)
    for event in pygame.event.get():
        if not hasattr(event, 'key'): continue
        if event.key == K_RIGHT:
            direction = RIGHT
            mode = '3rd person'
            got_input = True
        elif event.key == K_LEFT:
            direction = LEFT
            mode = '3rd person'
            got_input = True
        elif event.key == K_UP:
            direction = UP
            mode = '3rd person'
            got_input = True
        elif event.key == K_DOWN:
            direction = DOWN
            mode = '3rd person'
            got_input = True
        elif event.key == K_a:
            direction2 = LEFT
            mode = '3rd person'
            got_input = True
        elif event.key == K_s:
            direction2 = DOWN
            mode = '3rd person'
            got_input = True
        elif event.key == K_d:
            direction2 = RIGHT
            mode = '3rd person'
            got_input = True
        elif event.key == K_w:
            direction2 = UP
            mode = '3rd person'
            got_input = True
        elif event.key == K_r:
            trans = np.eye(4)
            mode = '1st person'
            K = CAMERA_MATRIX.copy()
            got_input = True
        elif event.key == K_SPACE:
            if y == 0:
                v = 1
            mode = '3rd person'
            got_input = True
        elif event.key == K_ESCAPE:
            pygame.display.quit()
            exit()
        elif event.key == K_c and event.type == pygame.KEYDOWN:
            frame_idx = max(0, smooth_factor*(frame_idx//smooth_factor - 1))
        elif event.key == K_v and event.type == pygame.KEYDOWN:
            frame_idx = min(len(RTs) - 1, smooth_factor*(frame_idx//smooth_factor + 1))
        elif (event.key == K_o or event.key == K_p or event.key == K_k or event.key == K_l) and event.type == pygame.KEYDOWN:
            print(K)
    x_pressed, z_pressed = pygame.key.get_pressed()[K_x],pygame.key.get_pressed()[K_z]
    if x_pressed and not z_pressed:
        frame_idx = min(len(RTs) - 1, frame_idx + 1)
    if not x_pressed and z_pressed:
        frame_idx = max(0, frame_idx - 1)
    if pygame.key.get_pressed()[K_o]:
        if pygame.key.get_pressed()[K_LSHIFT]:
            K[0, 0] -= 2
        else:
            K[0, 0] += 2
    if pygame.key.get_pressed()[K_p]:
        if pygame.key.get_pressed()[K_LSHIFT]:
            K[1, 1] -= 2
        else:
            K[1, 1] += 2
    if pygame.key.get_pressed()[K_l]:
        if pygame.key.get_pressed()[K_LSHIFT]:
            K[0, 2] -= 2
        else:
            K[0, 2] += 2
    if pygame.key.get_pressed()[K_k]:
        if pygame.key.get_pressed()[K_LSHIFT]:
            K[1, 2] -= 2
        else:
            K[1, 2] += 2
    x += vx
    z += vz
    move[0] += vx
    move[1] += v
    move[2] += vz
    a += va
    if direction == UP:
        dots[:, 1] = dots[:, 1] - 0.1
        lines[:, 1] = lines[:, 1] - 0.1
        move[1] -= 0.1
    elif direction == DOWN:
        dots[:, 1] = dots[:, 1] + 0.1
        lines[:, 1] = lines[:, 1] + 0.1
        move[1] += 0.01
    elif direction == RIGHT:
        vx -= 0.07
    elif direction == LEFT:
        vx += 0.07
    if direction2 == UP:
        vz -= 0.05
    elif direction2 == DOWN:
        vz += 0.05
    if direction2 == LEFT:
        va=1
    elif direction2 == RIGHT:
        va=-1
    if vz > 0:
        vz -= 0.005
    elif vz < 0:
        vz += 0.005
    if vx > 0:
        vx -= 0.01
    elif vx < 0:
        vx += 0.01
    if abs(vx) < 0.00001:
        vx = 0
    if abs(vz) < 0.00001:
        vz = 0
    if va < 0.1 and va > -0.1:
        va = 0
    elif va > 0:
        va -= 0.1
    elif va < 0:
        va += 0.1
    allign_y = False
    if y + v > 0:
        v -= 0.05
        y += v
    elif y + v < 0:
        v = 0
        allign_y = True

    if allign_y == True:
        trans = make_tranformation(trans,vx, y, vz, va)
        y = 0
    trans = make_tranformation(trans, vx, -v, vz, va)
    if mode == '1st person':
        draw_lines2(transform(lines, RTs[frame_idx].dot(trans)), K)
        draw_dots_with_color2(transform(dots, RTs[frame_idx].dot(trans)), dot_colors, K)
    else:
        draw_lines2(transform(lines, trans), K)
        draw_lines2(transform(camera_cube, trans.dot(np.linalg.inv(RTs[frame_idx]))), K, True)
        draw_dots_with_color2(transform(dots, trans), dot_colors, K)
    pygame.display.set_caption("frame = {}".format(match_frames_old.FRAMES_STEP * frame_idx / smooth_factor))
    pygame.display.flip()
