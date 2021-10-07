#!/usr/bin/env python3

import numpy as np
import cv2
import pickle
import os
"""
parameters = cv2.aruco.DetectorParameters_create()
corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
"""
class Calibrater:
    def __init__(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

        if not os.path.isfile('camera_parameters.pickle'):
            print("Couldn't open camera_parameters.pickle, please calibrate camera")
        else:
            params = pickle.load('camera_parameters.pickle')
            self.cam_mat = params['cameraMatrix']
            self.dist = params['dist']
            self.tvecs = params['tvecs']
            self.rvecs = params['rvecs']
        
    # returns the basis vectors and origo for the plane defined
    # by the fiducial markers.
    def find_plane(self, img):
        paramets = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=paramerts)
        rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 
                                                                  0.05, 
                                                                  self.cam_mat, 
                                                                  self.dist,
                                                                  self.rvecs,
                                                                  self.tvecs)

        



    def initialize_camera_parameters():
        pass

def main():
    calib = Calibrater()
    


if __name__ == '__main__':
    main()
