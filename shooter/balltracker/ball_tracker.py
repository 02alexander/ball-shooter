#!/usr/bin/env python3

import cv2
import numpy as np
from collections import deque
from .convex_hull import graham_scan
from ..utils import *

import time

def dist_to_shadow(color, pixel):
        cn = 1/np.sqrt(color[:,:,0]**2+color[:,:,1]**2+color[:,:,2]**2)
        for i in range(3):
            d = np.multiply(color[:,:,i],cn[:,:])
            color[:,:,i] = d
        
        dot = np.sum([ np.multiply(color[:,:,i], pixel[:,:,i]) for i in range(3)], axis=0)
        d = np.zeros(np.shape(color))
        for i in range(3):
            d[:,:,i] = dot
        dot = d        

        proj = np.multiply(color,dot)
        diff = proj-pixel
        norm = np.sqrt(diff[:,:,0]**2+diff[:,:,1]**2+diff[:,:,2]**2)
        return norm



# color and pixel are images now
def dist_to_shadow_vectorized(color, pixel):
    color = color / np.linalg.norm(color)
    stack = np.concatenate((color, pixel), axis=2)
    print(np.shape(stack))
    s = np.array([1,1,1])
    proj = np.apply_along_axis(lambda row : np.dot(row[:3], row[3:])*s, 2, stack)
    return np.linalg.norm(proj-pixel)

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
        all_outline, d = od

        
        """if ignore_rule is not None:
            outline = []
            for cord in all_outline:
                if not ignore_rule(cord):
                    outline.append(cord)
            outline = np.array(outline)
        else:
            outline = all_outline"""

        circ = circles(outline, 4)
        inframe_circs = []
        # adds all valid circles to inframe_circs
        for (center, radius) in circ:

            # checks if the center is not nan and also in the image
            if np.isnan(np.sum(center)) or np.isinf(np.sum(center)) or int(center[0]) < 0 or int(center[1]) < 0 or int(center[0]) >= h or int(center[1]) >= w:
                continue

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
        #return (best_center, r, outline)
    
    def bottom(self):
        cr = self.detect_circle()
        if cr is None:
            return None
        center, radius = cr
        return center+radius

    # ignore_rule is a function that says wheter a image coordinate 
    # is in a portion of an image that should be ignored.
    def outline(self, img, ignore_rule=None):
        d = abs_diff(self.reference_img, img)
        d = (255*(d>60)).astype(np.uint8)
        highlighted = highlight_pixels( cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR), d)
        
        edges = np.where(highlighted[:,:,0] == 255)
        if ignore_rule is not None:
            edges = edges[ignore_rule(edges)]
        
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
    cap = cv2.VideoCapture(0)
    while True:
        r, ref_img = cap.read()
        ref_img = cv2.GaussianBlur(ref_img, (7,7), 20)*1.0
        if r:
            break

    i = 0
    while True:
        ret, img = cap.read()
        img = cv2.GaussianBlur(img, (7,7), 20)*1.0
        #img = cv2.blur(img, (5,5))*1.0
        if not ret:
            continue
        #print(ref_img[0,0])
        #print(img[0,0])
        d = dist_to_shadow(np.copy(ref_img), np.copy(img))
        #d = cv2.blur(d, (5,5))

        print(d[0,0])
        print(i)
        i = i+1

        ball_img = (d/255.0 > 0.2)*1.0
        center_of_mass = mass_center(ball_img)
        print(center_of_mass)
        draw_rectangle(img, center_of_mass, 3)

        cv2.imshow("test", ball_img)
        cv2.imshow("window", img/255.0)
        k = cv2.waitKey(1)
        if k == 27:
            break
    
    """
    bt = BallTracker()
    while True:
        img_outline = np.zeros((bt.new_size[1], bt.new_size[0]))
        c = bt.detect_circle(ignore_rule=None)
        #c = bt.detect_circle()
        img = bt.last_frame
        
        d = abs_diff(bt.reference_img, cv2.blur(img, (5,5)))
        d = (255*(d>60)).astype(np.uint8)
        highlighted = highlight_pixels(img, d)

        if c is not None:
            center, _r, outline = c
            #print("outline=",outline)
            for cord in list(outline):
                img_outline[cord[0], cord[1]] = 255
            cord = np.array([center[0], center[1]])
            draw_rectangle(img, center, 2)

        cv2.imshow("outline", img_outline)
        cv2.imshow("highlighted", highlighted)
        cv2.imshow("window", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            bt.reset_reference()
        elif k == 27:
            break"""




if __name__ == "__main__":
    main()
