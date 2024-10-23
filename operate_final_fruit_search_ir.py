# teleoperate the robot, perform SLAM and object detection

# basic python packages
import numpy as np
import cv2 
import os, sys
import time
import torch
# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector

# for IR sensor
import time
DR = 16
DL = 19

class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = Alphabot(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.ckpt, use_gpu=False)
            self.network_vis = np.ones((240, 320,3))* 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')

    # wheel control
    def control(self, velo:dict=None, turn=None):
        if (velo != None):
            if (turn):
                print(f'CONTROL: TURNING\ttime={velo["time"]}')
                lv, rv = self.pibot.set_velocity(command=velo['velocity'], turning_tick=velo['turn_tick'], time=velo['time'])
            else:
                print(f'CONTROL: STRAIGHT\ttime={velo["time"]}')
                lv, rv = self.pibot.set_velocity(command=velo['velocity'], tick=velo['tick'], time=velo['time'])
        else:
            if args.play_data:
                lv, rv = self.pibot.set_velocity()            
            else:
                lv, rv = self.pibot.set_velocity(
                    self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        cv2.imshow('image', cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR))
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            # detector output :
            # w, h = 640 , 480
            # data = np.zeros((h, w))
            # data[int(np.round(ymin)):int(np.round(ymax)), int(np.round(xmin)):int(np.round(xmax))] = int(label) #label
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [3, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-3, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 3]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -3]
            # stop
            elif event.type != pygame.KEYDOWN:
                self.command['motion'] = [0, 0]

            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()
    
    # import RPi.GPIO as GPIO
    
    # #for IR sensor
    # def ir_sensor():
        # DR_status = GPIO.input(DR)
        # DL_status = GPIO.input(DL)
        # if(DL_status == 0):
            # IR_detect = 1
            # print("nak kena left liao")
        # elif (DR_status == 0):
            # IR_detect = 2
            # print("nak kena right liao")
        # else:
            # IR_detect = 0  
        # return IR_detect
                   
    

from generateMap import drive_to_point, get_fruit_list, get_gt_fruit_vec, a_star_algo, mesh_grid_plot
        
if __name__ == "__main__":
    ### PATH PLANNING
    DIST_PER_POINT = 0.1
    start_pose = [0, 0, 0]
    gt_tag, gt_vec, fruit_tag, fruit_vec = get_gt_fruit_vec('generated_map.txt')

    fruit_search = get_fruit_list('search_list.txt')
    step_list = a_star_algo(gt_vec, fruit_vec, 0.1, fruit_tag, fruit_search)

    point = [0, 0]
    waypoint_list = []
    for i in range(len(step_list)):
        point_list = []
        for j in range(len(step_list[i])):
            point[0] += step_list[i][j][0] * 0.1
            point[1] += -step_list[i][j][1] * 0.1
            point[0] = round(point[0], 2)
            point[1] = round(point[1], 2)
            point_list.append(point.copy())
        waypoint_list.append(point_list.copy())

    mesh_grid_plot(gt_tag, gt_vec, fruit_tag, fruit_vec, waypoint_list)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/last.pth')
    args, _ = parser.parse_known_args()

    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()
    
    ### GET FRUIT SEARCH LIST
    fruit_search = get_fruit_list('search_list.txt')
    i_fruit = 0
    j_step = 0
    curr_pose = [0, 0, 0]
    num_turn = 0
    num_turn_thres = 20
    num_straight = 0
    num_straight_thres = 5
    fruit_found = False
    obstacle_found = False
    moving_straight = False

    ### INITIALIZE OPERATE
    operate = Operate(args)
    operate.ekf_on = True
    operate.ekf.taglist = [i for i in range(1,11)]
    operate.ekf.markers = gt_vec
    num_markers = gt_vec.shape[1]
    operate.ekf.P = np.block(
        [[operate.ekf.P, np.zeros((3, 2*num_markers))],
        [np.zeros((2*num_markers, 2*num_markers + 3))]]
    )

    while True:
        operate.update_keyboard()

        # PERFORM FRUIT DETECTION
        operate.take_pic()
        operate.command['inference'] = True
        operate.detect_target()

        print(operate.detector.border)
        print(len(operate.detector.border))

        if i_fruit == (len(fruit_search)):
            print("FOUND ALL THREE FRUITS!!!")
            exit()

        print(f'FINDING FRUIT: {fruit_search[i_fruit]}')

        if (j_step) == 0:
            set_velo = drive_to_point(waypoint_list[i_fruit][-1], curr_pose)
            drive_meas = operate.control(set_velo[0], turn=True)
            time.sleep(0.2)     # SLP1
            drive_meas = operate.control(set_velo[1], turn=False)
            time.sleep(0.5)     # SLP2
            j_step += 1

        fruit_found = False
        obstacle_found = False
        # if at least one fruit fruit detected
        if len(operate.detector.border) > 0:
            # loop through each fruit detected bounding box
            for fruit_index in range(len(operate.detector.border)):
                # retrieve variable from bounding box
                xmin, ymin, xmax, ymax, label = operate.detector.border[fruit_index]
                xmiddle = xmin + (xmax-xmin)/2

                LABEL_STR = ['background', 'redapple', 'greenapple', 'orange', 'mango', 'capsicum', 'aruco']
                label_str = LABEL_STR[int(label)]

                # visualize the result
                print(f'FOUND Fruit{fruit_index}: {label_str}, xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}, xmiddle: {xmiddle}')


                # if fruit not matched, i.e. obstacles
                if label_str != fruit_search[i_fruit]:
                    if (moving_straight):
                        # if obstacle near
                        if (ymax > 450):
                            print("OBSTACLE NEAR")
                            obstacle_found = True
                            # obstacle at left
                            if (xmiddle < 200):
                                print("OBSTACLE AT LEFT!")
                                
                                # A1: TURN RIGHT (roughly 90 degrees)       ### TODO: TEST
                                velo = ({
                                    'velocity': [0, -1],
                                    'turn_tick': 15,
                                    'time': 0.36
                                })
                                drive_meas = operate.control(velo, turn=True)
                            # obstacle at right
                            elif (xmiddle > 440):
                                print("OBSTACLE AT RIGHT")
                                # A2: TURN LEFT (roughly 90 degrees)       ### TODO: TEST
                                velo = ({
                                    'velocity': [0, 1],
                                    'turn_tick': 15,
                                    'time': 0.4
                                })
                                drive_meas = operate.control(velo, turn=True)
                        
                            # A3: GO STRAIGHT after turning (roughly 1/3)       ### TODO: TEST
                            velo = ({
                                'velocity': [1, 0],
                                'tick': 20,
                                'time': 0.5,
                            })
                            drive_meas = operate.control(velo, turn=False)

                            # obstacle at left
                            if (xmiddle < 200):
                                # A1: TURN BACK RIGHT (roughly 90 degrees)       ### TODO: TEST
                                velo = ({
                                    'velocity': [0, -1],
                                    'turn_tick': 15,
                                    'time': 0.36
                                })
                                drive_meas = operate.control(velo, turn=True)
                            # obstacle at right
                            elif (xmiddle > 440):
                                # A2: TURN BACK LEFT (roughly 90 degrees)       ### TODO: TEST
                                velo = ({
                                    'velocity': [0, 1],
                                    'turn_tick': 15,
                                    'time': 0.4
                                })
                                drive_meas = operate.control(velo, turn=True)
                        
                            # A3: GO STRAIGHT after turning (roughly 1/3)       ### TODO: TEST
                            velo = ({
                                'velocity': [1, 0],
                                'tick': 20,
                                'time': 0.5,
                            })
                            drive_meas = operate.control(velo, turn=False)

            
                if label_str == fruit_search[i_fruit]:
                    print('FRUIT Check')

                    fruit_found = True
                    moving_straight = True
                    num_turn = 0
                    num_straight = 0
                    if (ymax) < 450:

                        if (xmiddle < 200):
                            print("FRUIT AT LEFT!")
                            velo = ({
                                'velocity': [0, 1],
                                'turn_tick': 15,
                                'time': 0.15
                            })
                            drive_meas = operate.control(velo, turn=True)
                        elif (xmiddle > 440):
                            print("FRUIT AT RIGHT")
                            velo = ({
                                'velocity': [0, -1],
                                'turn_tick': 15,
                                'time': 0.13
                            })
                            drive_meas = operate.control(velo, turn=True)
                    
                        velo = ({
                            'velocity': [1, 0],
                            'tick': 25,
                            'time': 0.5,
                        })
                        drive_meas = operate.control(velo, turn=False)

                    # else fruit near
                    else:
                        print("FRUIT NEAR!!!")
                        new_theta = np.arctan2(waypoint_list[i_fruit][-1][1]-curr_pose[1], waypoint_list[i_fruit][-1][0]-curr_pose[0])
                        curr_pose = waypoint_list[i_fruit][-1]
                        curr_pose.append(new_theta)
                        print(f'curr_pose:{curr_pose}')
                        i_fruit += 1
                        j_step = 0
                        time.sleep(5)   # SLP3

        if not(fruit_found):
            print("IR SENSORRR CHECK")
            print(operate.pibot.get_ir()[0])
            print(operate.pibot.get_ir()[1])
            print(f"NO MATCHED FRUIT, num_turn = {num_turn}")
            if (num_turn) < num_turn_thres:
                #  FRUIT DETECTION
                operate.take_pic()
                operate.command['inference'] = True
                operate.detect_target()
                velo = ({
                    'velocity': [0, -1],
                    'turn_tick': 20,
                    'time': 0.1
                })
                drive_meas = operate.control(velo, turn=True)
                num_turn += 1
                moving_straight = False
            #  straight
            else:
                moving_straight = True
                if (num_straight == 0):
                    velo = ({
                        'velocity': [0, 1],
                        'turn_tick': 20,
                        'time': 0.13
                    })
                    drive_meas = operate.control(velo, turn=True)

                    time.sleep(0.5)     # SLP4

                if (num_straight) < num_straight_thres:

                    velo = ({
                        'velocity': [1, 0],
                        'tick': 20,
                        'time': 0.5,
                    })
                    drive_meas = operate.control(velo, turn=False)
                    num_straight += 1
                else:
                    num_turn = 0
                    num_straight = 0

            time.sleep(0.1)     

        if (moving_straight == 1):
            print("IR   SENSORRR")
            print(operate.pibot.get_ir()[0])
            print(operate.pibot.get_ir()[1])
            # obstacle left
            if (operate.pibot.get_ir()[0] == False):
                
                time.sleep(0.5)

                velo = ({
                    'velocity': [-1, 0],
                    'tick': 20,
                    'time': 1.3,
                })
                drive_meas = operate.control(velo, turn=False)

                print('OBSTACLE AT LEFT! From IR Sensor')

                velo = ({
                    'velocity': [0, -1],
                    'turn_tick': 15,
                    'time': 0.36
                })
                drive_meas = operate.control(velo, turn=True)

                velo = ({
                    'velocity': [1, 0],
                    'tick': 20,
                    'time': 0.5,
                })
                drive_meas = operate.control(velo, turn=False)

                velo = ({
                    'velocity': [0, 1],
                    'turn_tick': 15,
                    'time': 0.4
                })
                drive_meas = operate.control(velo, turn=True)



            # obstacle right
            elif (operate.pibot.get_ir()[1]==False):
                print('OBSTACLE AT RIGHT! From IR Sensor')

                time.sleep(0.5)

                velo = ({
                    'velocity': [-1, 0],
                    'tick': 20,
                    'time': 1.3,
                })
                drive_meas = operate.control(velo, turn=False)

                velo = ({
                    'velocity': [0, 1],
                    'turn_tick': 15,
                    'time': 0.4
                })
                drive_meas = operate.control(velo, turn=True)

                velo = ({
                    'velocity': [1, 0],
                    'tick': 20,
                    'time': 0.5,
                })
                drive_meas = operate.control(velo, turn=False)

                velo = ({
                    'velocity': [0, -1],
                    'turn_tick': 15,
                    'time': 0.36
                })
                drive_meas = operate.control(velo, turn=True)



            moving_straight = False