import ast
import numpy as np
import matplotlib.pyplot as plt
from A_star import a_star_algorithm
import sys
import time
sys.path.insert(0, "util")
from pibot import Alphabot

def parse_groundtruth_fruit(fname : str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())
        
        aruco_dict = {}
        fruit_dict = {}
        # fruit_num = 1
        for key in gt_dict:
            if key.startswith("aruco"):
                aruco_num = int(key.strip('aruco')[:-2])
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
            else:
                fruit_dict[key[:-2]] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
                # fruit_num += 1

    return aruco_dict, fruit_dict

def get_vector(aruco0 : dict):
    points0 = []
    keys = []
    for key in aruco0:
        points0.append(aruco0[key])
        keys.append(key)
    return keys, np.hstack(points0)

def get_gt_fruit_vec(filename='M4_true_map.txt'):
    # filename='M4_true_map.txt'
    gt_aruco, fruit = parse_groundtruth_fruit(filename)

    gt_tag, gt_vec, fruit_tag, fruit_vec = None, None, None, None

    if gt_aruco != {}:
        gt_tag, gt_vec = get_vector(gt_aruco)
    if fruit:
        fruit_tag, fruit_vec = get_vector(fruit)

    return gt_tag, gt_vec, fruit_tag, fruit_vec

def mesh_grid_plot(gt_tag, gt_vec, fruit_tag, fruit_vec, waypoint_list):
    ax = plt.gca()
    ax.scatter(gt_vec[0,:], gt_vec[1,:], marker='o', color='C0', s=100)
    ax.scatter(fruit_vec[0,:], fruit_vec[1,:], marker='x', color='C1', s=100)
    for i in range(len(gt_tag)):
        ax.text(gt_vec[0,i]+0.05, gt_vec[1,i]+0.05, gt_tag[i], color='C0', size=12)
    for i in range(len(fruit_tag)):
        ax.text(fruit_vec[0,i]+0.05, fruit_vec[1,i]+0.05, fruit_tag[i], color='C1', size=12)
    col = ['r', 'g', 'b', 'c', 'm']
    mark = ['*', '+', 'p', '^', 'v']
    for i in range(len(waypoint_list)):
        waypoint_vec = [[], []]
        for j in range(len(waypoint_list[i])):
            waypoint_vec[0].append(waypoint_list[i][j][0])
            waypoint_vec[1].append(waypoint_list[i][j][1])
            # ax.scatter(waypoint_list[i][j][0], waypoint_list[i][j][1], marker=mark[i], color=col[i], s=100)
            ax.text(waypoint_list[i][j][0]+0.03, waypoint_list[i][j][1]+0.03, j+1, color=col[i], size=8)
        ax.scatter(waypoint_vec[0], waypoint_vec[1], marker=mark[i], color=col[i], s=70)
        # ax.text(waypoint_vec[0,:]+0.05, waypoint_vec[1,:], marker=mark[i], color=col[i], s=100)

    plt.title('Arena')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    plt.legend(['Aruco Marker','Fruit', 'First', 'Second', 'Third', 'Fourth', 'Fifth'])
    plt.grid()
    plt.show()

def get_fruit_list(fname):
    with open(fname) as file_in:
        fruits = []
        for fruit in file_in:
            fruits.append(fruit.strip('\n'))
        return fruits

def a_star_algo(gt_vec, fruit_vec, DIST_PER_POINT, fruit_tag, fruit_search):
    # DIST_PER_POINT = 0.4/4

    NUM_LIST = np.linspace(-1.2, 1.2, 7)
    round_gt_vec = np.copy(gt_vec)
    for idx, val in enumerate(gt_vec[0]):
        round_gt_vec[0][idx] = round(NUM_LIST[np.argmin(abs(NUM_LIST-val))], 2)
    for idx, val in enumerate(gt_vec[1]):
        round_gt_vec[1][idx] = round(NUM_LIST[np.argmin(abs(NUM_LIST-val))], 2)
    round_fruit_vec = np.copy(fruit_vec)
    for idx, val in enumerate(fruit_vec[0]):
        round_fruit_vec[0][idx] = round(NUM_LIST[np.argmin(abs(NUM_LIST-val))], 2)
    for idx, val in enumerate(fruit_vec[1]):
        round_fruit_vec[1][idx] = round(NUM_LIST[np.argmin(abs(NUM_LIST-val))], 2)

    # gt_arr = np.round(gt_vec/DIST_PER_POINT).astype(int)
    gt_arr = np.round(round_gt_vec/DIST_PER_POINT).astype(int)
    # fruit_arr = np.round(fruit_vec/DIST_PER_POINT).astype(int)
    fruit_arr = np.round(round_fruit_vec/DIST_PER_POINT).astype(int)

    fruit_search_arr = np.zeros((2,3), dtype=int)
    for idx, fruit in enumerate(fruit_search):
        fruit_idx = fruit_tag.index(fruit)
        fruit_search_arr[0][idx] = fruit_arr[0][fruit_idx]
        fruit_search_arr[1][idx] = fruit_arr[1][fruit_idx]

    obs_arr = np.array([np.append(gt_arr[0], fruit_arr[0]), np.append(gt_arr[1], fruit_arr[1])])

    step_list = a_star_algorithm(obstacles_list=obs_arr, fruit_list=fruit_search_arr)

    return step_list

def clamp_angle(rad_angle=0, min_value=-np.pi, max_value=np.pi):
    """
    Restrict angle to the range [min, max]
    :param rad_angle: angle in radians
    :param min_value: min angle value
    :param max_value: max angle value
    """

    if min_value > 0:
        min_value *= -1

    angle = (rad_angle + max_value) % (2 * np.pi) + min_value

    return angle

def drive_to_point(waypoint, robot_pose):
    ip = '192.168.137.68'
    port = '8000'
    ppi = Alphabot(ip, port)
    set_velo = []

    # imports camera / wheel calibration parameters
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')                # m / ticks
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')             # meter (distance between two wheels)

    ### INPUTS
    # waypoint = [x, y]
    # robot_pose = [x, y, theta]

    turn_vel = 13
    drive_vel = 20

    ### TURNING
    theta_r = robot_pose[2]                                                     # angle of robot from origin
    theta_w = np.arctan2(waypoint[1]-robot_pose[1], waypoint[0]-robot_pose[0])  # angle of waypoint from current position
    
    theta_off = np.radians(10)   # make it turn slightly more left
    theta_turn = clamp_angle(theta_w - theta_r - theta_off)                                 # angle to be turned by robot
    
    r = baseline/2
    d_turn = r * theta_turn

    # turn towards the waypoint
    turn_time = d_turn / (turn_vel*scale)           # DONE: CONVERT WHEEL VEL to CORRECT UNIT (TICKS/s OR RAD/s)
    turn_time = abs(turn_time)
    
    if (theta_turn) != 0:
        if (theta_turn > 0):        # TURN RIGHT
            set_velo.append({
                'velocity': [0, 1],
                'turn_tick': turn_vel,
                # 'time': turn_time[0],
                'time': turn_time,
            })
            # ppi.set_velocity([0, 1], turning_tick=turn_vel, time=turn_time)
        else:
            set_velo.append({       # TURN LEFT
                'velocity': [0, -1],
                'turn_tick': turn_vel,
                # 'time': turn_time[0],
                'time': turn_time,
            })
            # ppi.set_velocity([0, -1], turning_tick=turn_vel, time=turn_time)
    else:
        set_velo.append({
            'velocity': [0, 0],
            'turn_tick': 0,
            'time': 0,
        })

    # time.sleep(0.25)
    ### DRIVE STRAIGHT
    d_robot = np.sqrt(robot_pose[0]**2 + robot_pose[1]**2)
    d_waypoint = np.sqrt(waypoint[0]**2 + waypoint[1]**2)
    d_move = abs(d_waypoint - d_robot)                   # in meter
    d_move = np.sqrt( (waypoint[0]-robot_pose[0])**2 + (waypoint[1]-robot_pose[1])**2 )

    # after turning, drive straight to the waypoint
    drive_time = d_move / (drive_vel*scale)         # DONE: wheel_vel (tick/s) * scale (m/tick) = meter/s
    drive_time = min(drive_time,0.5)
    # print("Driving for {:.2f} seconds".format(drive_time))

    set_velo.append({
        'velocity': [1, 0],
        'tick': drive_vel,
        # 'time': drive_time[0],
        'time': drive_time,
    })
    # ppi.set_velocity([1, 0], tick=drive_vel, time=drive_time)
    ####################################################

    # print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

    # theta_new = theta_r + theta_turn
    # waypoint.append(theta_new)
    return set_velo


if __name__ == '__main__':
    DIST_PER_POINT = 0.4/4
    start_pose = [0, 0, 0]
    gt_tag, gt_vec, fruit_tag, fruit_vec = get_gt_fruit_vec('M4_true_map_new.txt')
    
    fruit_search = get_fruit_list('search_list.txt')

    step_list = a_star_algo(gt_vec, fruit_vec, DIST_PER_POINT, fruit_tag, fruit_search)
    # move list = [ [move_x1, move_y1], [move_x2, move_y2], ... ]
    # step list = [ move_list_fruit1, move_list_fruit2, ...]

    point = [0, 0]
    waypoint_list = []
    for i in range(len(step_list)):
        point_list = []
        for j in range(len(step_list[i])):
            point[0] += step_list[i][j][0] * DIST_PER_POINT
            point[1] += -step_list[i][j][1] * DIST_PER_POINT
            point_list.append(point.copy())
        waypoint_list.append(point_list.copy())

    mesh_grid_plot(gt_tag, gt_vec, fruit_tag, fruit_vec, waypoint_list)
    
    curr_pose = start_pose
    for i in range(len(waypoint_list)):
        for j in range(len(waypoint_list[i])):
            set_velo = drive_to_point(waypoint_list[i][j],curr_pose)
    # # waypoint = [0, 0]

    # step = 1

    # for i in range(len(step_list)): #len(step_list)
    #     for j in range(len(step_list[i])):
    #         move_x = step_list[i][j][0] * DIST_PER_POINT
    #         move_y = -step_list[i][j][1] * DIST_PER_POINT


    #         waypoint = [0, 0]
    #         waypoint[0] = round(curr_pose[0] + move_x, 1)
    #         waypoint[1] = round(curr_pose[1] + move_y, 1)
    #         if (step == 18):
    #             print("STEP18!!!")
    #         print(f'STEP:{step}\t\tfruit{i}:\tmove_x = {move_x}, move_y = {move_y},\tpose = {curr_pose}')
    #         curr_pose = drive_to_point(waypoint,curr_pose)
    #         step += 1
    #         time.sleep(0.3)
    #         # print(curr_pose)
    #     print(f"REACHED FRUIT {i}")
    #     time.sleep(3)
    

    # mesh_grid_plot(gt_tag, gt_vec, fruit_tag, fruit_vec)