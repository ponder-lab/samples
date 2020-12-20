#!/usr/bin/env python
"""
Capture image from RasPi Cam
-Prerequisites
 -Install OpenCV: pip install opencv-python
 -You may need libatlas to use python3: sudo apt install libatlas-base-dev
 -picamera is already installed if raspbian
"""

import io
import numpy as np
import cv2
import picamera
import picamera.array

camera = picamera.PiCamera(resolution=(640, 480), framerate=30)
buf = picamera.array.PiRGBArray(camera, size=(640, 480))

for frame in camera.capture_continuous(buf, format='bgr',
                                       use_video_port=True):

    cv2.imshow("img", frame.array)
    if cv2.waitKey(1) == 27:
        break

    buf.truncate(0)

camera.close()
