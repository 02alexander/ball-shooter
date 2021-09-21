#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import pathlib
import itertools

#def change_center(img, new_center):

def random_string(n):
    res = ''
    chars = [i for i in itertools.chain(range(ord('A'), ord('Z')+1), range(ord('0'), ord('9')+1))]
    for i in range(n):
        res += chr(np.random.choice(chars))
    return res


def main():
    n = 10

    cap = cv2.VideoCapture(0)

    while True:
        if len(sys.argv) < 2:
            dir_path = pathlib.Path('.')
        else:
            dir_path = pathlib.Path(sys.argv[1])

        _, img = cap.read()
        cv2.imshow("window", img)
        k = cv2.waitKey(1)
        if k == ord('a'):
            p = dir_path / (random_string(n)+'.png')
            cv2.imwrite(str(p),img)
        if k == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
