#!/usr/bin/env python3


import cv2
import numpy as np

board_size = (10, 7)
square_size = 100 # number of pixels per square

img = np.zeros((board_size[0]*square_size, board_size[1]*square_size))
for i in range(board_size[0]):
    for j in range(board_size[1]):
        if (i+j)%2==0:
            img[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 255*np.ones((square_size, square_size))

cv2.imwrite('checkerboard.png', img)