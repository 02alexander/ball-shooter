#!/usr/bin/env python3

import os
import termios
import time

# flags that work
# -parenb -parodd -cmspar cs8 hupcl -cstopb cread clocal -crtscts -ignbrk -brkint -ignpar -parmrk -inpck -istrip -inlcr -igncr icrnl ixon -ixoff -iuclc -ixany -imaxbel -iutf8 opost olcuc ocrnl onlcr -onocr -onlret -ofill -ofdel nl0 cr0 tab0 bs0 vt0 ff0 isig icanon iexten echo echoe echok -echonl -noflsh -xcase -tostop -echoprt echoctl echoke -flusho -extproc

class Controller:
    def __init__(self):
        self.fd = os.open("/dev/ttyACM0", os.O_RDWR)
        if self.fd == -1:
            self.fd = os.open("/dev/ttyACM1", os.O_RDWR)
        if self.fd == -1:
            raise NameError("Error opening terminal device")

        attr = termios.tcgetattr(self.fd)

        attr[1] = attr[1] & ~(termios.OPOST | termios.ONLCR | termios.CBAUD)
        attr[1] |= termios.B38400

        termios.tcsetattr(self.fd, termios.TCSAFLUSH, attr)
        self.file = os.fdopen(self.fd, "w")
        self.time_last_turned = time.time()

    def shoot(self):
        self.file.write('shoot\n')
    
    def turn(self, angle):
        start = time.time()
        if start - self.time_last_turned > 0.2:
            self.file.write(f'{angle:.4}'+'\n')
            end = time.time()
            print("time to send message ", (end-start))
            self.time_last_turned = end
