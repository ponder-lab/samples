#!/usr/bin/env python

import random
import numpy as np
import tensorflow as tf
import cv2

import matplotlib.pyplot as plt


seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None]
x_test = x_test[..., None]

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1)
    #validation_split=0.2

#flow = datagen.flow(x_train, y_train, batch_size=16, subset="training")
#flow = datagen.flow(x_train, y_train, batch_size=16, subset="validation")
flow = datagen.flow(x_train, y_train, batch_size=16)

plt.figure(figsize=(19.2, 10.8))
for i in range(16):
    x, y = flow.next()
    for j in range(16):
        plt.subplot(16, 16, i*16+j+1)
        plt.imshow(x[j, ..., 0])
        plt.xticks([]), plt.yticks([]), plt.title(y[j], x=-0.2, y=0.6)
plt.show()


