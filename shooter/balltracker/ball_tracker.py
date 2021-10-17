#!/usr/bin/env python3

import cv2
import numpy as np
from collections import deque
from .convex_hull import graham_scan
from ..utils import *



class BallTracker:
    def __init__(self, cap=None, new_size=(160,120), display_window=None):
        self.new_size = new_size
        if cap == None:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cap
        self.last_frame = None
        _, reference_img = self.cap.read()
        self.raw_last_frame = reference_img
        self.reference_img = cv2.blur(cv2.resize(reference_img, new_size), (5,5))
        self.display_window = display_window


    def detect_circle(self):
        _, frame = self.cap.read()
        self.raw_last_frame = frame
        frame = cv2.resize(frame, self.new_size)
        self.last_frame = frame
        frame = cv2.blur(frame, (5,5))
        w, h = self.new_size

        od = self.outline(frame)
        if od is None:
            return None
        outline, d = od
        circ = circles(outline, 6)
        inframe_circs = []
        # adds all valid circles to inframe_circs
        for (center, radius) in circ:

            # checks if the center is not nan and also in the image
            if np.isnan(np.sum(center)) or np.isinf(np.sum(center)) or int(center[0]) < 0 or int(center[1]) < 0 or int(center[0]) >= h or int(center[1]) >= w:
                continue
            #print("shape(highlighted)=", np.shape(highlighted))
            #print(highlighted[int(center[0]), int(center[1])], int(center[0]), int(center[1]))
            if (d[int(center[0]), int(center[1])] == np.array([0,0,0])).all():
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
        cr = self.detect_circle()
        if cr is None:
            return None
        center, radius = cr
        return center+radius

    def outline(self, img):
        d = abs_diff(self.reference_img, img)
        d = (255*(d>60)).astype(np.uint8)
        highlighted = highlight_pixels( cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR), d)
        edges = np.where(highlighted[:,:,0] == 255)
        
        edges = np.where(highlighted[:,:,0] == 255)
        if len(edges) > 0 and len(edges[0]) > 5:
            outline = graham_scan(edges)
            outline = np.array(outline)
            return outline, d
        return None

    def reset_reference(self, new_image=None):
        if new_image is None:
            _, new_image = self.cap.read()
            new_image = cv2.blur(cv2.resize(new_image, self.new_size), (5,5))
        self.reference_img = new_image


def main():
    bt = BallTracker()
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.namedWindow("highlighted", cv2.WINDOW_NORMAL)
    while True:
        c = bt.detect_circle()
        img = bt.last_frame
        
        d = abs_diff(bt.reference_img, cv2.blur(img, (5,5)))
        d = (255*(d>30)).astype(np.uint8)
        highlighted = highlight_pixels(img, d)

        if c is not None:
            center, _r = c
            cord = np.array([center[1], center[0]])
            print(cord)
            draw_rectangle(img, center, 2)

        cv2.imshow("highlighted", highlighted)
        cv2.imshow("window", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            bt.reset_reference()
        elif k == 27:
            break




if __name__ == "__main__":
    main()
