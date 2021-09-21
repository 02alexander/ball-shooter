#!/usr/bin/env python3

import cv2
import numpy as np

# rotations for a 3d vector
def rotation_matrix(angle, axis=0):
    if axis==0:
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis==1:
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis==2:
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

def rotate_z(vec, angle):
    return np.dot(rotation_matrix(angle, axis=2), vec)

def rotate_x(vec, angle):
    return np.dot(rotation_matrix(angle, axis=0), vec)

def rotate_y(vec, angle):
    return np.dot(rotation_matrix(angle, axis=1), vec)

def rotate_around_axis(v, angle, axis):
    
    anglex = np.arctan2(axis[2], axis[1])
    axis = rotate_x(axis, -anglex)
    v = rotate_x(v, -anglex)
    anglez = np.arctan2(axis[1], axis[0])
    v = rotate_z(v, -anglez)
    v = rotate_x(v, angle)
    v = rotate_x(rotate_z(v, anglez) ,anglex)
    return v

    #return np.cos(angle)*v + (1-np.cos(angle))*np.dot(axis,v)*axis+np.sin(angle)*np.cross(axis,v)

def shoot_angle(vball, pball, absvproj):
    phi = np.arctan2(pball[1], pball[0])
    A = np.linalg.norm(pball)*np.linalg.norm(absvproj)
    theta = np.arcsin( (pball[1]*vball[0]-pball[0]*vball[1])/A )-phi
    t = pball[0]/(absvproj*np.cos(theta)-vball[0])
    if np.isnan(t) or np.isinf(t):
        return None
    return -theta

class CameraModel:
    def __init__(self, height, angle, vertical_angle, horizontal_angle, screen_width=640, screen_height=480):
        self.height = height
        self.angle = angle*np.pi/180
        self.vertical_angle = vertical_angle*np.pi/180
        self.horizontal_angle = horizontal_angle*np.pi/180
        self.screen_width = screen_width
        self.screen_height = screen_height

    """
    def ground_position(self, screen_pos):
        screen_pos[1] = self.screen_height-screen_pos[1]
        cam_pos = np.array([0.0, 0.0, self.height])
        cam_dir = np.array([0.0, np.cos(self.angle), -np.sin(self.angle)])
        px = (screen_pos[0]-self.screen_width/2)/(self.screen_width/2)
        py = (screen_pos[1]-self.screen_height/2)/(self.screen_height/2)
    """
    
    def ground_position(self, screen_pos):
        cam_pos = np.array([0.0, 0.0, self.height])
        cam_dir = np.array([0.0, np.cos(self.angle), -np.sin(self.angle)])
        px = (screen_pos[0]-self.screen_width/2)/(self.screen_width/2)
        py = (screen_pos[1]-self.screen_height/2)/(self.screen_height/2)
        
        theta_z = np.arcsin(px*np.sin(self.horizontal_angle))
        theta_x = np.arcsin(py*np.sin(self.vertical_angle))
        a = rotate_x(cam_dir, -theta_x)
        rot_axis = rotate_x(a, np.pi/2)
        ball_dir = rotate_around_axis(a, theta_z, rot_axis)
        
        
        t = -(self.height/ball_dir[2])
        return ball_dir*t+cam_pos
    

if __name__ == "__main__":
    cam_model = CameraModel(16.5, 10, 24.6, 30.6, screen_width=160, screen_height=120)

    cv2.namedWindow('irl', cv2.WINDOW_NORMAL)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    x = 60
    while True:
        irl = np.zeros((100,100))
        img = np.zeros((120, 160))

        for y in np.arange(10, 150, step=10):
            y = int(y)
            img[x, y] = 255
            irl_pos = cam_model.ground_position(np.array([y,x]))
            print(irl_pos)
            irl_pos = irl_pos.astype(np.int32)
            irl_pos[0] += 50
            irl_pos[1] = 100-irl_pos[1]

            if irl_pos[0] >= 100 or irl_pos[0] < 0 or irl_pos[1] >= 100 or irl_pos[1] < 0:
                continue
            irl[irl_pos[1], irl_pos[0]] = 255

        cv2.imshow('img', img)
        cv2.imshow('irl', irl)
        k = cv2.waitKey()
        if k == 27:
            break
        elif k == ord('+'):
            x += 5
        elif k == ord('-'):
            x -= 5 



    """pos = cam_model.ground_position(np.array([80, 50]))

    pball = np.array([0, 5])
    vball = np.array([5,0])
    absvproj = np.linalg.norm(vball)
    #absvproj = 10
    angle = shoot_angle(vball, pball, absvproj)
    print(angle*180/np.pi)"""
    
        
    """v = np.array([1, 1, 0])
    axis = np.array([1, 1, 1])
    a = rotate_around_axis(v, 120*np.pi/180, axis)
    print(a)"""

    