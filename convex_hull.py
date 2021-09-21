#!/usr/bin/env python3


import numpy as np
from functools import cmp_to_key
import cv2

def cross_z(p0, p1, p2):
    return (p1[0]-p0[0])*(p2[1]-p0[1])-(p1[1]-p0[1])*(p2[0]-p0[0])
    #return 

def graham_scan(points):

    #cords = np.vstack((x, y)).transpose()
    xs, ys = points
    points = [ np.array([x, y]) for (x,y) in zip(xs,ys) ]

    idx = np.argmin([point[1] for point in points ])
    p0 = points.pop(idx)
    points.sort(key=cmp_to_key(lambda p1, p2: cross_z(p0, p1, p2)))
    hull = [p0, points[0]]
    for point in points[1:]:
        while len(hull) > 1 and cross_z(hull[-2], hull[-1], point) >= 0:
            hull.pop(-1)
        hull.append(point)

    return hull


if __name__ == '__main__':
    img_points = np.zeros((100,100), dtype=np.uint8)
    points = (np.random.choice(range(100), size=15, replace=False), np.random.choice(range(100), size=15, replace=False))
    img_points[points] = 255
    cv2.namedWindow('points', cv2.WINDOW_NORMAL)
    cv2.namedWindow('hull', cv2.WINDOW_NORMAL)
    hull = graham_scan(points)
    hull = np.array(hull)

    img_hull = np.zeros((100, 100), dtype=np.uint8)
    xs = hull[:,0]
    ys = hull[:,1]
    img_hull[(xs, ys)] = 255
    while True:
        cv2.imshow('points', img_points)
        cv2.imshow('hull', img_hull)
        k = cv2.waitKey(0)
        if k == 27:
            break
    