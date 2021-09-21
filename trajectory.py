#!/usr/bin/env python3

import cv2
import numpy as np
#from scipy import stats
from utils import *
from collections import deque


def trajectory():
    pass

# minimizes the distance betweeen the points and the line.
def gradient_descent(data_points, n, learning_rate=0.001, start_params=np.array([0, 0, 0], dtype=np.float64)):

    for _ in range(n):
        print(start_params)
        tot_grad = np.zeros((3), dtype=np.float64)
        for data_point in data_points:
            e = error(data_point, start_params[0], np.array([start_params[1], start_params[2]]))
            #print(e)
            #print(error_gradient(data_point, start_params[0], np.array([start_params[1], start_params[2]])))
            tot_grad += e*error_gradient(data_point, start_params[0], np.array([start_params[1], start_params[2]]))
        tot_grad /= len(data_points)
        tot_grad[0] *= 0.01
        #tot_grad[0] = 1/(1+np.exp(-0.1*tot_grad[0]))-0.5
        print(tot_grad[0])

        #print((0.1-np.exp(-3*tot_grad[0])))

        #tot_grad[0] = 0.00001*tot_grad[0]
        #tot_grad[0] = 0

        #print(tot_grad[0])
        start_params += -tot_grad*learning_rate
    return start_params

def error(data_point, theta, p):
    dp = data_point-p
    v = np.array([np.cos(theta), np.sin(theta)])
    lmbda = np.dot(v, dp)/(np.linalg.norm(v)**2)
    r = v*lmbda
    return np.linalg.norm(r-dp)

def error_gradient(data_point, theta, p):
    dp = data_point-p
    v = np.array([np.cos(theta), np.sin(theta)])
    r = v*(np.dot(v, dp))/(np.linalg.norm(v)**2)
    e = r-dp
    angle = np.arctan2(e[0], e[1])
    return np.array([np.linalg.norm(r), np.sin(angle), np.cos(angle)])

def outliers(data_points):
    outliers = []
    distances = []
    for i in range(1, len(data_points)):
        distances.append(np.linalg.norm(data_points[i]-data_points[i-1]))
    med = np.median(distances)

    last_uncertain = -1
    for i in range(len(distances)):
        is_outlier = False
        if distances[i] >= 5*med:
            if last_uncertain == i-1 or i==len(distances)-1:
                is_outlier = True
            last_uncertain = i
        if is_outlier:
            outliers.append(i)
    return outliers

if __name__ == "__main__":
    #print(error_gradient(np.array([3, 3]), np.pi/4-0.0001, np.array([0,0]) ) )
    data_points = np.array([ np.array([x+40+np.random.randint(0,3), 2*x+10+np.random.randint(0,3)]) for x in range(30)])
    #data_points = [ np.array([3.0, 3.0]), np.array([2.0, 2.1]), np.array([1.0, 0.9])]
    image = np.zeros((200, 200))
    """start_p0x = np.average(data_points[:,0])
    start_p0y = np.average(data_points[:,1])
    print(start_p0x, start_p0y)
    params = gradient_descent(data_points, 400, learning_rate=0.03, start_params=[0, start_p0x, start_p0y])
    print(error(data_points[0], params[0], np.array([params[1], params[2]])))
    
    vdir = np.array([ int(100*np.cos(params[0])), int(100*np.sin(params[0])) ])
    p0 = np.array([int(params[1]), int(params[2])])
    print(p0)
    print(vdir)"""
    for data_point in data_points:
        image[data_point[1], data_point[0]] = 1

    slope, intercept = 0, 100
    #slope, intercept, r_value, p_value, std_err = stats.linregress(data_points[:,0], data_points[:,1])

    angle = np.arctan(slope)
    p0 = np.array([0, intercept])

    #image[10, 20] = 1

    print(slope)
    print(intercept)
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    while True:
        #cv2.line(image, tuple(p0-vdir), tuple(p0+2*vdir), (1), thickness=1)
        cv2.line(image, (0, int(intercept)), (199, int(slope*200)+int(intercept)), (1), thickness=1)
        cv2.imshow('window', image)

        k = cv2.waitKey(0)
        if k == 27:
            break
    
