# evaluate the map generated by SLAM against the true map
import ast
import numpy as np
import math as m
import json
import matplotlib.pyplot as plt
from generateMap import get_gt_fruit_vec


def parse_groundtruth(fname : str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())
        
        aruco_dict = {}
        for key in gt_dict:
            if key.startswith("aruco"):
                aruco_num = int(key.strip('aruco')[:-2])
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
    return aruco_dict

def parse_user_map(fname : str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        aruco_dict = {}
        for (i, tag) in enumerate(usr_dict["taglist"]):
            if (tag <= 10):  #Ensure there is no ghost tag
                aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
    return aruco_dict

def robot_pose(fname:str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        x = usr_dict["ekf_pose"]["x"]
        y = usr_dict["ekf_pose"]["y"]
        theta = usr_dict["ekf_pose"]["theta"]
    return x,y,theta


def match_aruco_points(aruco0 : dict, aruco1 : dict):
    points0 = []
    points1 = []
    keys = []
    for key in aruco0:
        if not key in aruco1:
            continue
        
        points0.append(aruco0[key])
        points1.append(aruco1[key])
        keys.append(key)
    return keys, np.hstack(points0), np.hstack(points1)
    
def match_aruco_points2(aruco0 : dict):
    points0 = []
    keys = []
    for key in aruco0:
        
        points0.append(aruco0[key])
        keys.append(key)
    return keys, np.hstack(points0)

def solve_umeyama2d(points1, points2):
    # Solve the optimal transform such that
    # R(theta) * p1_i + t = p2_i

    assert(points1.shape[0] == 2)
    assert(points1.shape[0] == points2.shape[0])
    assert(points1.shape[1] == points2.shape[1])


    # Compute relevant variables
    num_points = points1.shape[1]
    mu1 = 1/num_points * np.reshape(np.sum(points1, axis=1),(2,-1))
    mu2 = 1/num_points * np.reshape(np.sum(points2, axis=1),(2,-1))
    sig1sq = 1/num_points * np.sum((points1 - mu1)**2.0)
    sig2sq = 1/num_points * np.sum((points2 - mu2)**2.0)
    Sig12 = 1/num_points * (points2-mu2) @ (points1-mu1).T

    # Use the SVD for the rotation
    U, d, Vh = np.linalg.svd(Sig12)
    S = np.eye(2)
    if np.linalg.det(Sig12) < 0:
        S[-1,-1] = -1
    
    # Return the result as an angle and a 2x1 vector
    R = U @ S @ Vh
    theta = np.arctan2(R[1,0],R[0,0])
    x = mu2 - R @ mu1

    return theta, x

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

def translate(x,y,tx,ty):
    return (x + tx, y + ty)

def rotate(x,y,angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return x * c - y * s, x * s + y * c

def trans_matrix(tx,ty,theta):
    
    T = np.eye(4)
    T[0,0] = np.cos(theta)
    T[0,1] = -np.sin(theta)
    T[1,0] = np.sin(theta)
    T[1,1] = np.cos(theta)
    T[0,3] = tx
    T[1,3] = ty
 
    T = np.linalg.inv(T)
    
    return T

def compute_rmse(points1, points2):
    # Compute the RMSE between two matched sets of 2D points.
    assert(points1.shape[0] == 2)
    assert(points1.shape[0] == points2.shape[0])
    assert(points1.shape[1] == points2.shape[1])
    num_points = points1.shape[1]
    residual = (points1-points2).ravel()
    MSE = 1.0/num_points * np.sum(residual**2)

    return np.sqrt(MSE)

# read in the object poses
def parse_map(fname: str) -> dict:
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline()) 
        redapple_gt, greenapple_gt, orange_gt, mango_gt, capsicum_gt = [], [], [], [], []
        
        # remove unique id of targets of the same type 
        for key in gt_dict:
            if key.startswith('redapple'):
                redapple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('greenapple'):
                greenapple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('mango'):
                mango_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('capsicum'):
                capsicum_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
    # if more than 1 estimation is given for a target type, only the first estimation will be used
    num_per_target = 1 # max number of units per target type. We are only use 1 unit per fruit type
    if len(redapple_gt) > num_per_target:
        redapple_gt = redapple_gt[0:num_per_target]
    if len(greenapple_gt) > num_per_target:
        greenapple_gt = greenapple_gt[0:num_per_target]
    if len(orange_gt) > num_per_target:
        orange_gt = orange_gt[0:num_per_target]
    if len(mango_gt) > num_per_target:
        mango_gt = mango_gt[0:num_per_target]
    if len(capsicum_gt) > num_per_target:
        capsicum_gt = capsicum_gt[0:num_per_target]

    return redapple_gt, greenapple_gt, orange_gt, mango_gt, capsicum_gt


# compute the Euclidean distance between each target and its closest estimation and returns the average over all targets
def compute_dist(gt_list, est_list):
    gt_list = gt_list
    est_list = est_list
    dist_av = 0
    dist_list = []
    dist = []
    for gt in gt_list:
        # find the closest estimation for each target
        for est in est_list:
            dist.append(np.linalg.norm(gt-est)) # compute Euclidean distance
        dist.sort()
        dist_list.append(dist[0]) # distance between the target and its closest estimation
        print(dist_list)
        print(gt_list)
        print(est_list)
        print(dist)
        #dist = []
    dist_av = sum(dist_list)/len(dist_list) # average distance
    return dist_av
    
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Matching the estimated map and the true map")
    # parser.add_argument("groundtruth", type=str, help="The ground truth file name.")
    # parser.add_argument("estimate", type=str, help="The estimate file name.")
    args = parser.parse_args()

    # gt_aruco = parse_groundtruth(args.groundtruth)
    us_aruco = parse_user_map('lab_output/slam.txt')

    # taglist, us_vec, gt_vec = match_aruco_points(us_aruco, gt_aruco)
    taglist, us_vec = match_aruco_points2(us_aruco)

    idx = np.argsort(taglist)
    taglist = np.array(taglist)[idx]
    us_vec = us_vec[:,idx]
    # gt_vec = gt_vec[:, idx] 

    _, _, fruit_tag, fruit_vec = get_gt_fruit_vec('lab_output/targets.txt')

    # theta, x = solve_umeyama2d(us_vec, gt_vec)
    # us_vec_aligned = apply_transform(theta, x, us_vec)
    
    # diff = gt_vec - us_vec_aligned
    # rmse = compute_rmse(us_vec, gt_vec)
    # rmse_aligned = compute_rmse(us_vec_aligned, gt_vec)
    
    print()
    print("The following parameters optimally transform the estimated points to the ground truth.")
    # print("Rotation Angle: {}".format(theta))
    # print("Translation Vector: ({}, {})".format(x[0,0], x[1,0]))
    
    print(fruit_vec)

    print()
    print("Number of found markers: {}".format(len(taglist)))
    # print("RMSE before alignment: {}".format(rmse))
    # print("RMSE after alignment:  {}".format(rmse_aligned))
    robot_x, robot_y, robot_theta = robot_pose('lab_output/robot_pose.txt')
    print("SA")
    print(us_vec)
    popo = us_vec
    ori_x = 0.1
    ori_y = 0.1
    ori_theta = np.radians(0)
    ori_theta = clamp_angle(ori_theta)
   

    tx = ori_x - robot_x
    ty = ori_y - robot_y

    theta = ori_theta - clamp_angle(robot_theta)
    
    # translation = trans_matrix(robot_x,robot_y,clamp_angle(robot_theta))
    user_input = np.zeros((4,10))
    user_input[0,:] = us_vec[0,:]
    #print(us_vec)
    user_input[1,:] = us_vec[1,:]
    user_input[3,:] = 1
    # print(user_input)
    # print(type(user_input))
    # print(type(translation))
    # print(type(us_vec[0]))

    # transformed = translation @ user_input
    # us_vec[0] = transformed[0]
    # us_vec[1] = transformed[1]
    
    #us_vec[0],us_vec[1] = translate(us_vec[0],us_vec[1],tx,ty)
    us_vec[0],us_vec[1] = rotate(us_vec[0],us_vec[1],45.4)
    us_vec[0],us_vec[1] = translate(us_vec[0],us_vec[1],1.05,-1.1)





    us_vec = np.array([np.append(us_vec[0], fruit_vec[0]), np.append(us_vec[1], fruit_vec[1])])
    robot_x, robot_y, robot_theta = robot_pose('lab_output/robot_pose.txt')
    print("SA")
    print(us_vec)
    popo = us_vec
    ori_x = 0.1
    ori_y = 0.1
    ori_theta = np.radians(0)
    ori_theta = clamp_angle(ori_theta)
   

    tx = ori_x - robot_x
    ty = ori_y - robot_y

    theta = ori_theta - clamp_angle(robot_theta)
    
    # translation = trans_matrix(robot_x,robot_y,clamp_angle(robot_theta))
    user_input = np.zeros((4,15))
    user_input[0,:] = us_vec[0,:]
    #print(us_vec)
    user_input[1,:] = us_vec[1,:]
    user_input[3,:] = 1
    # print(user_input)
    # print(type(user_input))
    # print(type(translation))
    # print(type(us_vec[0]))

    # transformed = translation @ user_input
    # us_vec[0] = transformed[0]
    # us_vec[1] = transformed[1]
    
    #us_vec[0],us_vec[1] = translate(us_vec[0],us_vec[1],tx,ty)
    #us_vec[0],us_vec[1] = rotate(us_vec[0],us_vec[1],90)
    
    # NUM_LIST = np.linspace(-1.2, 1.2, 7)
    # round_us_vec = np.copy(us_vec)
    # for idx, val in enumerate(us_vec[0]):
    #     round_us_vec[0][idx] = round(NUM_LIST[np.argmin(abs(NUM_LIST-val))], 2)

    # for idx, val in enumerate(us_vec[1]):
    #     round_us_vec[1][idx] = round(NUM_LIST[np.argmin(abs(NUM_LIST-val))], 2)
    
  ##############################################
    import argparse

    parser = argparse.ArgumentParser("Matching the estimated map and the true map")
    parser.add_argument("--truth", default="final_true_map.txt", help="The ground truth file name.")
    parser.add_argument("--est", default="lab_output/targets.txt", help="The estimate file name.")
    args, _ = parser.parse_known_args()

    # read in ground truth and estimations
    redapple_gt, greenapple_gt, orange_gt, mango_gt, capsicum_gt = parse_map(args.truth)
    #print(redapple_gt)
    redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = parse_map(args.est)
    #print(redapple_est)
    #print("GREEN")
    #print(greenapple_gt)
    #print("GREEN EST")

    #print(greenapple_est)
    #print("GIIH")
    # compute average distance between a target and its closest estimation
    redapple_dist = compute_dist(redapple_gt,redapple_est)
    greenapple_dist = compute_dist(greenapple_gt,greenapple_est)
    orange_dist = compute_dist(orange_gt, orange_est)
    mango_dist = compute_dist(mango_gt, mango_est)
    capsicum_dist = compute_dist(capsicum_gt, capsicum_est)
    
    av_dist = (redapple_dist+greenapple_dist+orange_dist+mango_dist+capsicum_dist)/5
    
    print("Average distances between the targets and the closest estimations:")
    print("redapple = {}, greenapple = {}, orange = {}, mango = {}, capsicum = {}".format(redapple_dist,greenapple_dist,orange_dist,mango_dist,capsicum_dist))
    print("estimation error: ", av_dist)
    print(us_vec)
    

    file = open('generated_map.txt', 'w')
    file.write('{')
    print("FRUIT")
    print(fruit_tag)
    for i in range(10):
        # strWrite = f'"aruco{i+1}_0": {{"x": {us_vec[0][i]},"y": {us_vec[1][i]}}},'
        file.write(f'"aruco{i+1}_0": {{"x": {us_vec[0][i]},"y": {us_vec[1][i]}}},')
    for j in range(5):
        if (j == 4):
            file.write(f'"{fruit_tag[j]}_0": {{"x": {us_vec[0][10+j]},"y": {us_vec[1][10+j]}}}')
        else:
            file.write(f'"{fruit_tag[j]}_0": {{"x": {us_vec[0][10+j]},"y": {us_vec[1][10+j]}}},')
    file.write('}')
    file.close()
    
    
#####################################################
    

    
    print()
    # print('%s %7s %9s %7s %11s %9s %7s' % ('Marker', 'Real x', 'Pred x', 'Δx', 'Real y', 'Pred y', 'Δy'))
    # print('-----------------------------------------------------------------')
    # for i in range(len(taglist)):
        # print('%3d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f\n' % (taglist[i], gt_vec[0][i], us_vec_aligned[0][i], diff[0][i], gt_vec[1][i], us_vec_aligned[1][i], diff[1][i]))
    
    ax = plt.gca()
    # ax.scatter(gt_vec[0,:], gt_vec[1,:], marker='o', color='C0', s=100)
    # ax.scatter(us_vec_aligned[0,:], us_vec_aligned[1,:], marker='x', color='C1', s=100)
    ax.scatter(us_vec[0,:-5], us_vec[1,:-5], marker='x', color='C1', s=100)
    # ax.scatter(round_us_vec[0,:-5], round_us_vec[1,:-5], marker='x', color='blue', s=100)
    col = ['red', 'green', 'cyan', 'brown', 'pink']
    for k in range(5):
        ax.scatter(us_vec[0,10+k], us_vec[1,10+k], marker='^', color=col[k], s=100)
        ax.text(us_vec[0,10+k]+0.05, us_vec[1,10+k]+0.05, fruit_tag[k], color=col[k], size=12)
    for i in range(len(taglist)):
        ax.text(us_vec[0,i]+0.05, us_vec[1,i]+0.05, taglist[i], color='C1', size=12)
    plt.title('Arena')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    plt.legend(['Transformed',fruit_tag[0],fruit_tag[1],fruit_tag[2],fruit_tag[3],fruit_tag[4]])
    plt.grid()
    plt.show()