#!/usr/bin/env python3

import os
import termios
import time
import numpy as np
import cv2
from shooter.camcal.cammodel import CamModel
from shooter.balltracker.ball_tracker import BallTracker
from shooter.utils import draw_rectangle, kmeans

# flags that work
# -parenb -parodd -cmspar cs8 hupcl -cstopb cread clocal -crtscts -ignbrk -brkint -ignpar -parmrk -inpck -istrip -inlcr -igncr icrnl ixon -ixoff -iuclc -ixany -imaxbel -iutf8 opost olcuc ocrnl onlcr -onocr -onlret -ofill -ofdel nl0 cr0 tab0 bs0 vt0 ff0 isig icanon iexten echo echoe echok -echonl -noflsh -xcase -tostop -echoprt echoctl echoke -flusho -extproc

class Controller:
    def __init__(self, cammodel, shooter_pos):
        self.fd = os.open("/dev/ttyACM0", os.O_RDWR)
        if self.fd == -1:
            self.fd = os.open("/dev/ttyACM1", os.O_RDWR)
        if self.fd == -1:
            raise NameError("Error opening terminal device")

        attr = termios.tcgetattr(self.fd)

        attr[1] = attr[1] & ~(termios.OPOST | termios.ONLCR | termios.CBAUD)
        attr[1] |= termios.B9600

        #termios.tcsetattr(self.fd, termios.TCSAFLUSH, attr)
        self.file = os.fdopen(self.fd, "w")
        self.time_last_turned = time.time()

        self.cammodel = cammodel
        self.shooter_pos = shooter_pos

    def get_ball_angle(self, ball_img_cord):
        pos = self.cammodel.img_point_to_pos(ball_img_cord)
        shooter_dir = - self.shooter_pos
        #print(pos-self.shooter_pos)
        #print(shooter_dir)
        angle = np.arccos(np.dot(pos-self.shooter_pos, shooter_dir)/(np.linalg.norm(pos-self.shooter_pos)*np.linalg.norm(shooter_dir)))
        return angle
    

    def shoot(self):
        self.file.write('shoot\n')
    
    def turn(self, angle):
        start = time.time()
        if start - self.time_last_turned > 0.5:
            self.file.write(f'{angle:.4}'+'\n')
            end = time.time()
            print("time to send message ", (end-start))
            self.time_last_turned = end

def main():
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cammodel = CamModel(frame)
    bt = BallTracker(cap)
    
    cntrl = Controller(cammodel, np.array([0, 0.35]))

    while True:
        frame = bt.raw_last_frame
        circle = bt.detect_circle()
        if circle is None:
            print("no circle")
        else:
            center, _radius = circle
            center = np.array(center)
            center = np.flip(center)
            angle = cntrl.get_ball_angle(center*4)
            draw_rectangle(frame, np.flip(center)*4, 2)
            cntrl.turn(angle*180/np.pi)
            print(angle)
        
        for pos in cammodel._marker_positions:
            cord = cammodel.irl_cord_to_img_point(pos)
            draw_rectangle(frame, cord, 5)
            cord = cammodel.irl_cord_to_img_point(cammodel.origo)
            draw_rectangle(frame, cord, 5)
        
        cv2.imshow("window", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord('r'):
            bt.reset_reference()

if __name__ == "__main__":
    main()