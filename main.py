#!/usr/bin/env python3

import cv2
import numpy as np
from collections import deque
from shooter.utils import *
#from scipy import stats
from shooter.controller.controller import Controller
from shooter.balltracker.ball_tracker import BallTracker


if __name__ == "__main__":

    cv2.namedWindow('diff', cv2.WINDOW_NORMAL)
    cv2.namedWindow('irl', cv2.WINDOW_NORMAL)
    cv2.namedWindow('bottoms', cv2.WINDOW_NORMAL)

    new_size = (160, 120)
    bt = BallTracker(new_size=new_size)

    #cam_model = CameraModel(6.5, 16.3, 22, 26, screen_width=new_size[0], screen_height=new_size[1])
    #cam_model = CameraModel(16.5, 10, 23.96, 30.65, screen_width=new_size[0], screen_height=new_size[1])
    bottoms = deque(maxlen=20)
    controller = Controller()

    tstep = 0

    while True:

        # used for drawing purposes
        ground_img = np.zeros((100, 100), np.uint8)
        bottoms_img = np.zeros((120, 160, 3), np.uint8)

        bottom = bt.bottom()
        if bottom is not None:
            #irl_cord = cam_model.ground_position(np.array([bottom[1], bottom[0]]))
            bottoms.append(bottom)
        
        """outliers_indices = outliers(bottoms)
        
        outliers_indices.sort()
        
        trimmed_end = 0
        while trimmed_end < len(outliers_indices) and outliers_indices[trimmed_end] == trimmed_end:
            trimmed_end += 1
        trimmed_start = 0
        while trimmed_start < len(outliers_indices) and outliers_indices[trimmed_start] == trimmed_start:
            trimmed_start += 1

        # how many timesteps there are between the first and the last point in cleaned_bottoms
        time_steps = len(bottoms)-trimmed_end-trimmed_start
        

        cleaned_bottoms = list(bottoms.copy())
        for i in range(len(outliers_indices)):
            cleaned_bottoms.pop(outliers_indices[i]-i)
        """


        print(bottom)

        """
        if len(cleaned_bottoms) > 1:
            data_points = np.array(cleaned_bottoms)
            slope, intercept = 0, 100
            #slope, intercept, r_value, p_value, std_err = stats.linregress(data_points[:,1], data_points[:,0])
            
            minx = np.min(data_points[:,1])
            maxx = np.max(data_points[:,1])

            start = np.array([int(minx), int(slope*minx+intercept)])
            end = np.array([int(maxx), int(slope*maxx+intercept)])
            irl_start = cam_model.ground_position(np.array([minx, slope*minx+intercept]))
            irl_start_img = np.array([50+int(irl_start[0]), 100-int(irl_start[1])])
            irl_end = cam_model.ground_position(np.array([maxx, maxx*slope+intercept]))
            irl_end_img = np.array([50+int(irl_end[0]), 100-int(irl_end[1])])
            
            cv2.line(bottoms_img, start, end, 255,1)
            cv2.line(ground_img, irl_start_img.astype(int), irl_end_img.astype(int), 255, 1)

            vx = (irl_start[0]-irl_end[0])/(time_steps/48.0)
            vy = (irl_start[1]-irl_end[1])/(time_steps/48.0)
            v = np.array([vx, vy])
            #print("velocity", np.linalg.norm(v))
            print(irl_end)
            cur_angle = np.arctan2(irl_end[1], irl_end[0])
            print((cur_angle*180/np.pi-90))
            controller.turn((cur_angle*180/np.pi-90))
        """
        
        cv2.imshow('bottoms', bottoms_img)
        cv2.imshow('irl', ground_img)

        #for point in points:
        #    draw_rectangle(d, point, 3)
        #cv2.imshow("window", d)
        k = cv2.waitKey(1)

        if k == 27:
            break
        elif k == 114:
            bt.reset_reference()

        #prev = frame
    
    cv2.destroyAllWindows()
