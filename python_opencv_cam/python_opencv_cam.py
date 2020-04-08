#!/usr/bin/env python

import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
    quit()

while True:
    ret, img = cap.read()
    if not ret:
        break

    cv2.imshow("img_in", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

