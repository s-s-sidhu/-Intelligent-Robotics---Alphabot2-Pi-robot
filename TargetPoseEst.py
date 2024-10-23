# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
import cv2
import math
from machinevisiontoolbox import Image

import matplotlib.pyplot as plt
import PIL

# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(target_number, image_path):
    image = PIL.Image.open(image_path).resize((640,480), PIL.Image.NEAREST)
    target = Image(image)==target_number
    blobs = target.blobs()
    [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
    width = abs(u1-u2)
    height = abs(v1-v2)
    center = np.array(blobs[0].centroid).reshape(2,)
    box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "An image should contain only one object of each target type"
    return box

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most five types of targets in each image
    target_lst_box = [[], [], [], [], []]
    target_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}

    # add the bounding box info of each target in each image
    # target labels: 1 = redapple, 2 = greenapple, 3 = orange, 4 = mango, 5=capsicum, 0 = not_a_target
    img_vals = set(Image(base_dir / file_path, grey=True).image.reshape(-1))
    for target_num in img_vals:
        if target_num > 0:
            try:
                box = get_bounding_box(target_num, base_dir/file_path) # [x,y,width,height]
                pose = image_poses[file_path] # [x, y, theta]
                target_lst_box[target_num-1].append(box) # bouncing box of target
                target_lst_pose[target_num-1].append(np.array(pose).reshape(3,)) # robot pose
            except ZeroDivisionError:
                pass

    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i+1] = {'target': box, 'robot': pose}
        
    return completed_img_dict

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = []
    redapple_dimensions = [0.074, 0.074, 0.087]
    target_dimensions.append(redapple_dimensions)
    greenapple_dimensions = [0.081, 0.081, 0.067]
    target_dimensions.append(greenapple_dimensions)
    orange_dimensions = [0.075, 0.075, 0.072]
    target_dimensions.append(orange_dimensions)
    mango_dimensions = [0.113, 0.067, 0.058] # measurements when laying down
    target_dimensions.append(mango_dimensions)
    capsicum_dimensions = [0.073, 0.073, 0.088]
    target_dimensions.append(capsicum_dimensions)

    target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num-1][2]
        
        ######### Replace with your codes #########
        # TODO: compute pose of the target based on bounding box info and robot's pose
        # This is the default code which estimates every pose to be (0,0)
        target_pose = {'x': 0.0, 'y': 0.0}

        #define the variables here
        height = float(box[3]) #height of the object pixel on image which is b
        theta = float(robot_pose[2])
        width = float(box[2]) #width of the object pixel on image which is a

        u0 = camera_matrix[0][2]

        target_z = focal_length * true_height / height #true height is the coordinate y 

        target_x = (float(box[0]-u0)) * target_z / focal_length

        alpha = theta - np.arctan(target_x / target_z)


        dist = np.sqrt(target_x**2 + target_z**2)


        camera_dist = 0.0255555
        target_pose['x'] = float(robot_pose[0]) + target_z*np.cos(theta) + target_x*np.sin(theta) + camera_dist*np.cos(theta)
        target_pose['y'] = float(robot_pose[1]) + target_z*np.sin(theta) - target_x*np.cos(theta) + camera_dist*np.sin(theta)

        # target_pose['x'] = float(robot_pose[0]) + dist*np.cos(alpha) 
        # target_pose['y'] = float(robot_pose[1]) + dist*np.sin(alpha) 
        
        target_pose_dict[target_list[target_num-1]] = target_pose
        ###########################################
    
    return target_pose_dict

# merge the estimations of the targets so that there are at most 1 estimate for each target type
def merge_estimations(target_map):
    print("111")
    target_map = target_map
    redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = [], [], [], [], []
    target_est = {}
    num_per_target = 1 # max number of units per target type. We are only use 1 unit per fruit type
    # combine the estimations from multiple detector outputs
    for f in target_map:
        print("f")
        print(f)
        for key in target_map[f]:
            print(key)
            if key.startswith('redapple'):
                print("Enter1")

                redapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('greenapple'):
                print("Enter2")

                greenapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                print("Enter3")

                orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('mango'):
                print("Enter4")

                mango_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('capsicum'):
                print("Enter5")

                capsicum_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    # TODO: the operation below is the default solution, which simply takes the first estimation for each target type.
    # Replace it with a better merge solution.
    print("EST")
    num_per_target = 0
    if len(redapple_est) > num_per_target:
        redapple_est = np.mean(redapple_est,axis = 0) 
    print(redapple_est)
    if len(greenapple_est) > num_per_target:
        greenapple_est = np.mean(greenapple_est,axis = 0) 
        #print("MEANFREEN")
    print(greenapple_est)

    if len(orange_est) > num_per_target:
        orange_est = np.mean(orange_est,axis = 0)  
    print(orange_est)

    if len(mango_est) > num_per_target:
        mango_est = np.mean(mango_est,axis = 0)
    print(mango_est)

    if len(capsicum_est) > num_per_target:
        capsicum_est = np.mean(capsicum_est,axis = 0)
    print(capsicum_est)
    print("EST END")
    num_per_target = 1

    # print(redapple_est)

    for i in range(num_per_target):
        print(redapple_est)
        try:
            target_est['redapple_'+str(i)] = {'x':redapple_est[0], 'y':redapple_est[1]}
        except:
            pass
        try:
            target_est['greenapple_'+str(i)] = {'x':greenapple_est[0], 'y':greenapple_est[1]}
        except:
            pass
        try:
            target_est['orange_'+str(i)] = {'x':orange_est[0], 'y':orange_est[1]}
        except:
            pass
        try:
            target_est['mango_'+str(i)] = {'x':mango_est[0], 'y':mango_est[1]}
        except:
            pass
        try:
            target_est['capsicum_'+str(i)] = {'x':capsicum_est[0], 'y':capsicum_est[1]}
        except:
            pass
    ###########################################
        
    return target_est


if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    
    
    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    print("HI")
    print(image_poses)

    # estimate pose of targets in each detector output
    target_map = {}        
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)
    print("HI1")
    print(target_map)
    # merge the estimations of the targets so that there are only one estimate for each target type
    target_est = merge_estimations(target_map)
    print("HI")
    print(target_est) 

#     image = cv2.imread('lab_output/pred_0.png')
#     print(np.unique(image))
# # show the image, provide window name first
#     cv2.imshow('image window', image)
# # add wait key. window waits until user presses a key
#     cv2.waitKey(0)
# # and finally destroy/close all open windows
#     cv2.destroyAllWindows()

    #cv2.imshow('image','lab_output/pred_0.png')
                     
    # save target pose estimations
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo)
    
    # print(target_map)
    print('Estimations saved!')