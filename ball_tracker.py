#!/usr/bin/env python3

import cv2
import numpy as np
from aimer import CameraModel, shoot_angle
from collections import deque
from convex_hull import graham_scan
from trajectory import outliers
from utils import *
#from scipy import stats
from controller import Controller

def channel_diffs(img):
    a = np.abs(img[:,:,2].astype(np.uint32)-img[:,:,1].astype(np.uint32))
    b = np.abs(img[:,:,1].astype(np.uint32)-img[:,:,0].astype(np.uint32))
    return np.dstack((a,b, np.zeros(a.shape))).astype(np.uint8)
    #return np.dstack((np.full(a.shape, 255), np.full(a.shape, 128), np.full(a.shape, 0))).astype(np.uint8)

# where pixels is an image where every non-black pixel is highlighted in img and everything else becomes black.
def highlight_pixels(img, pixels):
    m = np.average(pixels, axis=2)>0
    indicator = np.dstack((m, m, m))
    return np.multiply(img, indicator)

def kmeans(img, k=3):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    critera = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, critera, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

class BallTracker:
    def __init__(self, new_size=(160,120)):
        self.new_size = new_size
        self.cap = cv2.VideoCapture(0)
        self.last_frame = None
        _, reference_img = self.cap.read()
        self.reference_img = cv2.blur(cv2.resize(reference_img, new_size), (5,5))

    def circle(self):
        _, frame = self.cap.read()
        frame = cv2.resize(frame, new_size)
        self.last_frame = frame
        frame = cv2.blur(frame, (5,5))
        w, h = self.new_size

        outline = self.outline(frame)
        if outline is None:
            return None
        circ = circles(outline, 6)
        inframe_circs = []

        # adds all valid circles to inframe_circs
        for (center, radius) in circ:

            # checks if the center is not nan and also in the image
            if np.isnan(center[0]) or np.isnan(center[1]) or int(center[0]) < 0 or int(center[1]) < 0 or int(center[0]) >= h or int(center[1]) >= w:
                continue
            inframe_circs.append((center, radius))
        circ = inframe_circs
        if len(circ) == 0:
            return None
        r = None
        best_center = real_center([center for (center, radius) in circ])
        for (center, radius) in circ:
            if np.linalg.norm(best_center-center) < 0.000001:
                r = radius
        return (best_center, r)
    
    def bottom(self):
        cr = self.circle()
        if cr is None:
            return None
        center, radius = cr
        return center+radius

    def outline(self, img):
        d = abs_diff(self.reference_img, img)
        cv2.imshow('diff', d)
        d = (255*(d>40)).astype(np.uint8)
        highlighted = highlight_pixels( cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR), d)
        edges = np.where(highlighted[:,:,0] == 255)
        
        edges = np.where(highlighted[:,:,0] == 255)
        if len(edges) > 0 and len(edges[0]) > 5:
            outline = graham_scan(edges)
            outline = np.array(outline)
            return outline
        return None

    def reset_reference(self, new_image=None):
        if new_image is None:
            _, new_image = self.cap.read()
            new_image = cv2.blur(cv2.resize(new_image, self.new_size), (5,5))
        self.reference_img = new_image

if __name__ == "__main__":

    cv2.namedWindow('diff', cv2.WINDOW_NORMAL)
    cv2.namedWindow('irl', cv2.WINDOW_NORMAL)
    cv2.namedWindow('bottoms', cv2.WINDOW_NORMAL)

    new_size = (160, 120)
    bt = BallTracker(new_size=new_size)

    #cam_model = CameraModel(6.5, 16.3, 22, 26, screen_width=new_size[0], screen_height=new_size[1])
    cam_model = CameraModel(16.5, 10, 23.96, 30.65, screen_width=new_size[0], screen_height=new_size[1])
    bottoms = deque(maxlen=20)
    controller = Controller()

    tstep = 0

    while True:

        # used for drawing purposes
        ground_img = np.zeros((100, 100), np.uint8)
        bottoms_img = np.zeros((120, 160, 3), np.uint8)

        bottom = bt.bottom()
        if bottom is not None:
            irl_cord = cam_model.ground_position(np.array([bottom[1], bottom[0]]))
            bottoms.append(bottom)
        outliers_indices = outliers(bottoms)
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

        irl_cord = None
        for bottom in cleaned_bottoms:
            draw_rectangle(bottoms_img, bottom.astype(int), 1)
            irl_cord = cam_model.ground_position(np.array([bottom[1], bottom[0]]))
            if np.isnan(irl_cord[0]) or np.isnan(irl_cord[1]):
                continue
            irl_cord_int = np.array([int(irl_cord[0])+50, 100-int(irl_cord[1])])
            if irl_cord_int[0] < 0 or irl_cord_int[1] < 0 or irl_cord_int[0] >= 100 or irl_cord_int[1] >= 100:
                continue
            ground_img[irl_cord_int[1], irl_cord_int[0]] = 255

        print(bottom)
        print(irl_cord)

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
