#!/usr/bin/env python3

import os
import termios
import time
import numpy as np
import cv2
from collections import deque
from shooter.camcal.cammodel import CamModel
from shooter.balltracker.ball_tracker import BallTracker
from shooter.utils import draw_rectangle, kmeans, abs_diff, highlight_pixels

# flags that work
# -parenb -parodd -cmspar cs8 hupcl -cstopb cread clocal -crtscts -ignbrk -brkint -ignpar -parmrk -inpck -istrip -inlcr -igncr icrnl ixon -ixoff -iuclc -ixany -imaxbel -iutf8 opost olcuc ocrnl onlcr -onocr -onlret -ofill -ofdel nl0 cr0 tab0 bs0 vt0 ff0 isig icanon iexten echo echoe echok -echonl -noflsh -xcase -tostop -echoprt echoctl echoke -flusho -extproc

class Controller:
    def __init__(self, cammodel, shooter_pos, queuelen=30, projectile_velocity=3.34, fire_delay=0.36):
        self.fd = os.open("/dev/ttyACM0", os.O_RDWR)
        if self.fd == -1:
            self.fd = os.open("/dev/ttyACM1", os.O_RDWR)
        if self.fd == -1:
            raise NameError("Error opening terminal device")

        attr = termios.tcgetattr(self.fd)
        attr[1] = attr[1] & ~(termios.ONOCR | termios.ONLCR | termios.OCRNL | termios.OLCUC)
        #attr[1] |= termios.B9600
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, attr)

        self.file = os.fdopen(self.fd, "w")
        self.time_last_turned = time.time()

        self.cammodel = cammodel
        
        # indexed by id
        self.aruco_markers_img_cord = [
            np.reshape(cammodel.irl_cord_to_img_point(np.transpose(cammodel.aruco_tvecs[np.where(cammodel.aruco_ids==i)[0][0]])), (2)) for i in range(4)
        ]

        self.shooter_pos = shooter_pos

        self.previous_positions = deque(maxlen=queuelen) # list of [(position, time)]
        self.projectile_velocity = projectile_velocity
        self.fire_delay = fire_delay
        self.is_loaded = True

    def ignore_rule(self, cord):
        v = self.aruco_markers_img_cord[0]-self.aruco_markers_img_cord[3]
        a = np.cross(v, self.aruco_markers_img_cord[2]-self.aruco_markers_img_cord[3])
        b = np.cross(v, cord-self.aruco_markers_img_cord[3])
        return a*b < 0

    def get_ball_angle(self, ball_img_cord):
        pos = self.cammodel.img_point_to_pos(ball_img_cord)
        shooter_dir = - self.shooter_pos
        v = pos-self.shooter_pos
        #angle = np.arccos(np.dot(pos-self.shooter_pos, shooter_dir)/(np.linalg.norm(pos-self.shooter_pos)*np.linalg.norm(shooter_dir)))
        cross = np.cross(v, shooter_dir)
        angle = np.arcsin(cross/(np.linalg.norm(v)*np.linalg.norm(shooter_dir)))
        return angle
    
    def run_iteration(self, bt, window=True):
        frame = bt.raw_last_frame
        circle = bt.detect_circle(ignore_rule=lambda cord : self.ignore_rule(cord))
        if circle is None:
            #print("no circle detected")
            pass
        else:
            center, _radius = circle
            center = np.array(center)
            center = np.flip(center)
            angle = self.get_ball_angle(center*4)
            
            if window is not None:
                draw_rectangle(frame, np.flip(center)*4, 2)
            
            self.turn(angle*180/np.pi)
            #print(angle)

            pos = self.cammodel.img_point_to_pos(center*4)
            self.previous_positions.append((pos, time.time()))
        
        if len(self.previous_positions) == self.previous_positions.maxlen and self.is_loaded:
            velocity = self.estimate_velocity()
            
            if np.linalg.norm(velocity) < 0.1:
                print("FIRE!")
                self.shoot()
                self.is_loaded = False
            


        if window is not None:
            for pos in self.cammodel._marker_positions:
                cord = self.cammodel.irl_cord_to_img_point(pos)
                draw_rectangle(frame, cord, 5)
                cord = self.cammodel.irl_cord_to_img_point(self.cammodel.origo)
                draw_rectangle(frame, cord, 5)
            cv2.imshow(window, frame)

    def estimate_velocity(self):
        if len(self.previous_positions) <= 2:
            return None
        else:
            (start_pos, start_time) = self.previous_positions[0]
            (end_pos, end_time) = self.previous_positions[-1]
            return (start_pos-end_pos)/(end_time-start_time)

    def shoot(self):
        self.file.write('shoot\n')

    def turn(self, angle):
        start = time.time()
        if start - self.time_last_turned > 0.001:
            self.file.write(f'{angle:.4}'+'\n')
            end = time.time()
            #print("time to send message ", (end-start))
            self.time_last_turned = end

def main():
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.namedWindow("highlighted", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cammodel = CamModel(frame)
    bt = BallTracker(cap)
    
    cntrl = Controller(cammodel, np.array([0, 0.35]))

    while True:
        cntrl.run_iteration(bt, window="window")
        
        img = bt.last_frame
        d = abs_diff(bt.reference_img, cv2.blur(img, (5,5)))
        d = (255*(d>30)).astype(np.uint8)
        highlighted = highlight_pixels(img, d)
        cv2.imshow("highlighted", highlighted)

        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord('r'):
            bt.reset_reference()

if __name__ == "__main__":
    main()