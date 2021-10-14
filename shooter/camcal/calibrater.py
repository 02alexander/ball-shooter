#!/usr/bin/env python3

import numpy as np
import cv2
import pickle
import os
from ..utils import draw_rectangle

"""
parameters = cv2.aruco.DetectorParameters_create()
corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
"""
class Calibrater:
    def __init__(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        this_dir = os.path.dirname(__file__) 
        if not os.path.isfile(this_dir+'/camera_parameters.pickle'):
            #TODO prompt user to take picture of checkerboard
            print("Couldn't open "+this_dir+"/camera_parameters.pickle, please calibrate camera")
        else:
            params = pickle.load(open('shooter/camera_calibration/camera_parameters.pickle', 'rb'))
            print(params)
            self.cam_mat = params['cameraMatrix']
            self.dist = params['dist']
            self.tvecs = params['tvecs']
            self.rvecs = params['rvecs']
        


def main():
    pass


if __name__ == '__main__':
    main()
