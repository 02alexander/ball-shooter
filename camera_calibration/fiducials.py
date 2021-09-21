#!/usr/bin/env python3

import cv2
import sys
import numpy as np
import pickle
from collections import deque

# kan man fylla i ett rutigt papper som marker?
"""
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

markers = [cv2.aruco.drawMarker(aruco_dict, i, 100) for i in range(4)]

m = 255*np.ones((565,400))
m[0:markers[0].shape[0], 0:markers[0].shape[1]] = markers[0]
m[-markers[1].shape[0]:, 0:markers[1].shape[1]] = markers[1]
m[-markers[2].shape[0]:, -markers[2].shape[1]:] = markers[2]
m[0:markers[3].shape[0], -markers[3].shape[1]:] = markers[3]

print(np.shape(m))
print(m[0:20, 0:20])

cv2.imwrite("fiducials.png", m)
cv2.imshow("window", m)
k = cv2.waitKey()"""

parameters = pickle.load(open('camera_parameters.pickle', 'rb'))
print(parameters)
print(parameters.keys())
cameraMatrix = parameters['cameraMatrix']
dist = parameters['dist']
paramrvecs = np.array(parameters['rvecs'])
paramtvecs = np.array(parameters['tvecs'])

cap = cv2.VideoCapture(0)

backup_rvecs = None
backup_tvecs = None

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()


    # Check if frame is not empty
    if not ret:
        continue

    # Convert from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 
                                                                  0.05, 
                                                                  cameraMatrix, 
                                                                  dist,
                                                                  paramrvecs,
                                                                  paramtvecs)
    
    if len(rvecs) != 4:
        if backup_tvecs is None or backup_rvecs is None:
            continue
        rvecs = backup_rvecs
        tvecs = backup_tvecs
    else:
        backup_rvecs = rvecs
        backup_tvecs = tvecs
    
    for i in range(len(rvecs)):
        rvec = rvecs[i]
        tvec = tvecs[i]
        #print()
        #print(ids[i])
        #print(tvec*100)
        #print(rvec)
        frame = cv2.aruco.drawAxis(frame, cameraMatrix, dist, rvec, tvec, 0.1)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

vecs = {'rvec': rvecs, 'tves': tvecs}
pickle.dump(vecs, open('vecs.pickle', 'wb'))

cv2.destroyAllWindows()