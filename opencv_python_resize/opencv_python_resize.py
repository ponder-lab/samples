import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

import cv2

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

img_size = (4, 4)

x_train_s = np.empty((x_train.shape[0],) + img_size, x_train.dtype)
for i in range(x_train.shape[0]):
    x_train_s[i] = cv2.resize(x_train[i], (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train_s[i], cmap=plt.cm.binary)
    plt.xlabel(str(y_train[i]))
plt.show()
