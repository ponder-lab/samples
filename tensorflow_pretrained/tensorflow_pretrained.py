#!/usr/bin/env python

import numpy as np
import cv2
import tensorflow.keras as keras

model = keras.applications.resnet50.ResNet50(weights="imagenet")
model.summary()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera")
    quit()

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = img[:, 80:560]

    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.resnet50.preprocess_input(x)

    preds = model.predict(x)

    preds_decoded = keras.applications.resnet50.decode_predictions(preds, top=3)
    for i in range(3):
        if preds_decoded[0][i][2] > 0.3:
            msg = preds_decoded[0][i][1] + "," + str(preds_decoded[0][i][2])
            cv2.putText(img, msg, (15, 15*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("img_in", img)

    #import code
    #code.InteractiveConsole(locals=locals()).interact()
    
    key = cv2.waitKey(1)
    if key == 27:
        break

