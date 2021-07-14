from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from pid_control import *

# Parameters
KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
AREA_WIDTH = 30.0  # potential area width [m]
# the number of previous positions used to check oscillations
OSCILLATIONS_DETECTION_LENGTH = 3

show_animation = True

class Observation:
    def __init__(self, start_pos=[0, 0], goal_pos=[0,0], grid_size=0.5, robot_radius=5, obstacle_pos_x=[1], obstacle_pos_y=[1]):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        self.obstacle_pos_x = obstacle_pos_x
        self.obstacle_pos_y = obstacle_pos_y


def calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy):
    minx = min(min(ox), sx, gx) - AREA_WIDTH / 2.0
    miny = min(min(oy), sy, gy) - AREA_WIDTH / 2.0
    maxx = max(max(ox), sx, gx) + AREA_WIDTH / 2.0
    maxy = max(max(oy), sy, gy) + AREA_WIDTH / 2.0
    xw = int(round((maxx - minx) / reso))
    yw = int(round((maxy - miny) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        x = ix * reso + minx

        for iy in range(yw):
            y = iy * reso + miny
            ug = calc_attractive_potential(x, y, gx, gy)
            uo = calc_repulsive_potential(x, y, ox, oy, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(ox):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - ox[minid], y - oy[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0


def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


def oscillations_detection(previous_ids, ix, iy):
    previous_ids.append((ix, iy))

    if (len(previous_ids) > OSCILLATIONS_DETECTION_LENGTH):
        previous_ids.popleft()

    # check if contains any duplicates by copying into a set
    previous_ids_set = set()
    for index in previous_ids:
        if index in previous_ids_set:
            return True
        else:
            previous_ids_set.add(index)
    return False


def potential_field_planning(sx, sy, gx, gy, ox, oy, reso, rr):

    # calc potential field
    pmap, minx, miny = calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy)
    path_ = []

    # search path
    d = np.hypot(sx - gx, sy - gy)
    ix = round((sx - minx) / reso)
    iy = round((sy - miny) / reso)
    gix = round((gx - minx) / reso)
    giy = round((gy - miny) / reso)

    if show_animation:
        draw_heatmap(pmap)
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(ix, iy, "*k")
        plt.plot(gix, giy, "*m")

    rx, ry = [sx], [sy]
    motion = get_motion_model()
    previous_ids = deque()

    while d >= reso:
        minp = float("inf")
        minix, miniy = -1, -1
        for i, _ in enumerate(motion):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(pmap) or iny >= len(pmap[0]) or inx < 0 or iny < 0:
                p = float("inf")  # outside area
                print("outside potential!")
            else:
                p = pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
        ix = minix
        iy = miniy
        xp = ix * reso + minx
        yp = iy * reso + miny
        d = np.hypot(gx - xp, gy - yp)
        rx.append(xp)
        ry.append(yp)
        path_.append([ix, iy])

        if (oscillations_detection(previous_ids, ix, iy)):
            print("Oscillation detected at ({},{})!".format(ix, iy))
            break

        if show_animation:
            plt.plot(ix, iy, ".r")
            plt.pause(0.01)

    print("Goal!!")

    return path_,rx, ry


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def path_tracking(path_, show_animation):
    ax = np.zeros(len(path_))
    ay = np.zeros(len(path_))
    for i in range(len(path_)):
        ax[i] = path_[i][0]
        ay[i] = path_[i][1]


    goal = [ax[-1], ay[-1]]

    reference_path = CubicSplinePath(ax, ay)
    s = np.arange(0, reference_path.length, 0.1)

    t, x, y, yaw, v, goal_flag = simulate(reference_path, goal, path_)

    # Test
    assert goal_flag, "Cannot goal"

    if show_animation:  # pragma: no cover
        plt.plot(x, y, "-g", label="tracking")


def main(observation):
    print("potential_field_planning start")

    sx = observation.start_pos[0]  # start x position [m]
    sy = observation.start_pos[1]  # start y positon [m]
    gx = observation.goal_pos[0]  # goal x position [m]
    gy = observation.goal_pos[1]  # goal y position [m]
    grid_size = observation.grid_size  # potential grid size [m]
    robot_radius = observation.robot_radius  # robot radius [m]

    ox = observation.obstacle_pos_x  # obstacle x position list [m]
    oy = observation.obstacle_pos_y  # obstacle y position list [m]

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    # path generation
    path_,rx, ry = potential_field_planning(
        sx, sy, gx, gy, ox, oy, grid_size, robot_radius)

    #print(path_)

    #path tracking, pid_control
    path_tracking(path_, show_animation)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    observation = Observation()
    observation.start_pos = [0.0, 10.0] # start position [m]
    observation.goal_pos = [30.0, 30.0] # goal position [m]
    observation.grid_size = 0.5  # potential grid size [m]
    observation.robot_radius = 5.0 #robot radius [m]
    observation.obstacle_pos_x = [15.0, 5.0, 20.0, 25.0]  # obstacle x position list [m]
    observation.obstacle_pos_y = [25.0, 15.0, 26.0, 25.0]  # obstacle x position list [m]
    main(observation)
    print(__file__ + " Done!!")
