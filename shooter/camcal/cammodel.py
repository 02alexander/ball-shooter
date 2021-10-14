import numpy as np
import cv2
import pickle
import os
from ..utils import draw_rectangle

class CamModel:
    def __init__(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        this_dir = os.path.dirname(__file__) 
        if not os.path.isfile(this_dir+'/camera_parameters.pickle'):
            print("Couldn't open "+this_dir+"/camera_parameters.pickle, please calibrate camera")
        else:
            params = pickle.load(open(this_dir+'/camera_parameters.pickle', 'rb'))
            print(params)
            self.cam_mat = params['cameraMatrix']
            self.dist = params['dist']
            self.tvecs = params['tvecs']
            self.rvecs = params['rvecs']
        
    # returns the basis vectors and origo for the plane defined
    # by the fiducial markers.
    # returns (basis vector x, basis vector y, origo)
    def find_plane(self, img):
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=parameters)
        rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 
                                                                  0.05, 
                                                                  self.cam_mat, 
                                                                  self.dist,
                                                                  np.array(self.rvecs),
                                                                  np.array(self.tvecs))

        t = []
        for tvec in tvecs:
            t.append(np.array(tvec).reshape(3,1))
        tvecs = t

        midpoint = (tvecs[np.where(ids == 1)[0][0]] + tvecs[np.where(ids == 2)[0][0]] )*0.5
        midpoint = midpoint.reshape(3)
        basx = tvecs[np.where(ids==1)[0][0]] - tvecs[np.where(ids==2)[0][0]]
        basy = tvecs[np.where(ids==0)[0][0]] - tvecs[np.where(ids==1)[0][0]]
        basx = basx.reshape(3)
        basy = basy.reshape(3)
        basx = basx / np.sqrt(np.dot(basx,basx))
        basy = basy / np.sqrt(np.dot(basy,basy))
    
        return (basx, basy, midpoint, tvecs)


    # returns the position point in coordinate system described by origo, basx, basy where basx and basy is a ON-basis
    # point = [x, y]    the pixel coordinates
    def img_point_to_pos(self, origo, basx, basy, point):
        point = list(point)
        point.append(1)
        point = np.array(point)
        
        #m = self.cam_mat
        #vdir = np.array([point[0]-m[0,2], point[1]-m[1,2], 540])
        vdir = np.dot(np.linalg.inv(self.cam_mat), point)

        plane_normal = np.cross(basx, basy)
        D = np.dot(plane_normal, origo)
        t = D/(np.dot(vdir, plane_normal))
        
        # where vdir intersects the plane
        intersection_point = vdir*t

        x = np.dot(basx, intersection_point-origo[0])
        y = np.dot(basy, intersection_point-origo[1])

        return np.array([x, y])

    def irl_cord_to_img_point(self, irl):
        v = np.dot(self.cam_mat, irl)
        v = v/v[2]
        return np.array([v[1], v[0]]).round().astype('int32')

def reverse_tvecs(tvecs, rvecs):
    tvs = []
    for i in range(np.shape(tvecs)[0]):
        rm,_ = cv2.Rodrigues(rvecs[i])
        rm = np.transpose(rm)
        tv = np.dot(-rm, np.transpose(tvecs[i]))
        tvs.append(tv)
    return tvs


def main():
    cap = cv2.VideoCapture(0)    
    calib = CamModel()
    _, frame = cap.read()
    basx, basy, origo, marker_positions = calib.find_plane(frame)
    print(basx)
    print(basy)
    print("origo=",origo)
    print(calib.irl_cord_to_img_point(origo))
    print(np.dot(basx,basy))

    def print_pos(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(calib.img_point_to_pos(origo, basx, basy, np.array([x,y])))

    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("window", print_pos)

    while True:
        ret, frame = cap.read()

        for pos in marker_positions:
            cord = calib.irl_cord_to_img_point(pos)
            draw_rectangle(frame, cord, 5)
            cord = calib.irl_cord_to_img_point(origo)
            draw_rectangle(frame, cord, 5)

        #frame.draw_rectangle()
        cv2.imshow("window", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break



if __name__ == '__main__':
    main()