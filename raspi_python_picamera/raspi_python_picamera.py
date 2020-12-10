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

camera = picamera.PiCamera(resolution=(640, 480), framerate=30)

#camera.start_preview()

stream = io.BytesIO()
for _ in camera.capture_continuous(stream, format='raw',
                                   use_video_port=True):

    buf = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    buf = buf.reshape(-1, 640)

    img = cv2.cvtColor(buf, cv2.COLOR_YUV2BGR_I420)

    stream.seek(0)
    stream.truncate()

    cv2.imshow("img", img)
    if cv2.waitKey(1) == 27:
        break

#camera.stop_preview()
camera.close()
