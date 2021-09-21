#!/usr/bin/env python3

import numpy as np
import cv2

def linregress(x, y):
    pass

def abs_diff(img1, img2):
    a = np.array(img1).astype(np.int32)
    b = np.array(img2).astype(np.int32)
    return np.abs(a-b).astype(np.uint8)


def draw_rectangle(img, point, r):
    rows, cols, _ = img.shape
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            if i+point[0]<0 or i+point[0]>= rows or j+point[1] < 0 or j+point[1] >= cols:
                continue
            img[i+point[0]][j+point[1]] = [255, 0, 0]

def mass_center(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    x = np.fromfunction(lambda i, j: i, img.shape)
    y = np.fromfunction(lambda i, j: j, img.shape)
    xr = np.multiply(x, img)
    yr = np.multiply(y, img)

    x_avg = np.sum(xr)/np.sum(img)
    y_avg = np.sum(yr)/np.sum(img)
    return (int(x_avg), int(y_avg))


def intersection_point(v, p0, u, p1):
    scoef = u[1]-u[0]*v[1]/v[0]
    s = ( p0[1] + v[1]*(p1[0]-p0[0])/v[0] - p1[1])/scoef
    return u*s+p1

def circle_from_points(p0,p1,p2):
    v = p1-p0
    u = p2-p1
    vrot90 = np.array([-v[1], v[0]])
    urot90 = np.array([-u[1], u[0]])
    center = intersection_point(vrot90, p0+v*0.5, urot90, p1+u*0.5)
    radius = np.linalg.norm(center-p0)
    return (center, radius)

def circles(hull, n):
    circles = []
    for i in range(len(hull)):
        circles.append(circle_from_points( hull[i],hull[(i+n)%len(hull)],hull[(i+2*n)%len(hull)] ) )
    return circles

# finds the distance between p and the line defined by v and p0
def distance_from_line(v, p0, p):
    vrot90 = np.array([-v[1], v[0]])
    ip = intersection_point(v, p0, vrot90, p)
    return np.linalg.norm(ip-p)

def real_center(centers):
    best_center = None
    best_dist = None
    for center in centers:
        dist_sum = 0
        for center2 in centers:
            if np.linalg.norm(center2-center) < 0.0000001:
                continue
            dist_sum += 1/(np.linalg.norm(center2-center)**3)
        if best_dist == None or dist_sum > best_dist:
            best_center = center
            best_dist = dist_sum
    return best_center