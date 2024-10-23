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

from generateMap import drive_to_point, get_fruit_list, get_gt_fruit_vec, a_star_algo, mesh_grid_plot
        
if __name__ == "__main__":

    DIST_PER_POINT = 0.4/4
    start_pose = [0, 0, 0]
    gt_tag, gt_vec, fruit_tag, fruit_vec = get_gt_fruit_vec('M4_true_map_new.txt')
    
    fruit_search = get_fruit_list('search_list.txt')

    step_list = a_star_algo(gt_vec, fruit_vec, DIST_PER_POINT, fruit_tag, fruit_search)

    point = [0, 0]
    waypoint_list = []
    for i in range(len(step_list)):
        point_list = []
        for j in range(len(step_list[i])):
            point[0] += step_list[i][j][0] * DIST_PER_POINT
            point[1] += -step_list[i][j][1] * DIST_PER_POINT
            point[0] = round(point[0], 2)
            point[1] = round(point[1], 2)
            point_list.append(point.copy())
        waypoint_list.append(point_list.copy())

    mesh_grid_plot(gt_tag, gt_vec, fruit_tag, fruit_vec, waypoint_list)

    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.68')
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

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)
    operate.ekf_on = True
    operate.ekf.taglist = [i for i in range(1,11)]
    operate.ekf.markers = gt_vec
    num_markers = gt_vec.shape[1]
    operate.ekf.P = np.block(
        [[operate.ekf.P, np.zeros((3, 2*num_markers))],
        [np.zeros((2*num_markers, 2*num_markers + 3))]]
    )

    i_fruit = 0
    j_waypoint = 0
    k_turn = True
    set_velo = None

    while start:
        operate.update_keyboard()

        # ### RUN SLAM ###
        # n_observed_markers = len(operate.ekf.taglist)
        # if n_observed_markers == 0:
        #     if not operate.ekf_on:
        #         operate.notification = 'SLAM is running'
        #         operate.ekf_on = True
        #     else:
        #         operate.notification = '> 2 landmarks is required for pausing'
        # elif n_observed_markers < 3:
        #     operate.notification = '> 2 landmarks is required for pausing'
        # else:
        #     if not operate.ekf_on:
        #         operate.request_recover_robot = True
        #     operate.ekf_on = not operate.ekf_on
        #     if operate.ekf_on:
        #         operate.notification = 'SLAM is running'
        #     else:
        #         operate.notification = 'SLAM is paused'
        # ### RUN SLAM ###

        operate.take_pic()
        # curr_pose = [0,0,0]
        curr_pose = operate.ekf.robot.state
        print(f'CURRENTPOSE={curr_pose[0][0]},{curr_pose[1][0]},{curr_pose[2][0]}')

        if (k_turn):
            set_velo = drive_to_point(waypoint_list[i_fruit][j_waypoint], curr_pose)
            print(f'fruit={i_fruit}, step={j_waypoint:>2}-{j_waypoint+1:<2},\t waypoint={waypoint_list[i_fruit][j_waypoint]}')
            # print(f'fruit={i_fruit}, step={j_waypoint:<2}-{j_waypoint+1:<2},\t turn={k_turn},\t set_velo={set_velo[0]}')
            print(f'fruit={i_fruit}, step={j_waypoint:>2}-{j_waypoint+1:<2},\t turn={k_turn},\t comm={set_velo[0]["velocity"]},\t ttick={set_velo[0]["turn_tick"]}, time={set_velo[0]["time"]}')
        else:
            # print(f'fruit={i_fruit}, step={j_waypoint:<2}-{j_waypoint+1:<2},\t turn={k_turn},\t set_velo={set_velo[1]}\n')
            print(f'fruit={i_fruit}, step={j_waypoint:>2}-{j_waypoint+1:<2},\t turn={k_turn},\t comm={set_velo[1]["velocity"]},\t stick={set_velo[1]["tick"]}, time={set_velo[1]["time"]}\n')

        if not(k_turn):
            if ( j_waypoint < (len(waypoint_list[i_fruit])-1) ):
                j_waypoint += 1
            else:
                j_waypoint = 0
                if ( i_fruit < (len(waypoint_list)-1) ):
                    i_fruit += 1
                else:
                    print(f"END, i_fruit={i_fruit}, j_waypoint={j_waypoint}")
                    break


        if (k_turn):        # IF turning
            drive_meas = operate.control(set_velo[0], k_turn)
            k_turn = False  # NEXT go straight
        else:               # IF go straight
            drive_meas = operate.control(set_velo[1], k_turn)
            k_turn = True   # NEXT turn

        # curr_pose = operate.update_slam(drive_meas)
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        time.sleep(3)
