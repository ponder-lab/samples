#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import cv2
import matplotlib.pyplot as plt

def convert_to_gray(imgs):
    imgs_g = np.empty(imgs.shape[0:3], x_train.dtype)
    for i in range(imgs.shape[0]):
        imgs_g[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
    return imgs_g

def plot_sample(imgs_g, labels):
    for i in range(25):
        plt.subplot(5, 5, i+1)
        if imgs_g.ndim == 4:
            plt.imshow(imgs_g[i])
        else:
            plt.imshow(imgs_g[i], cmap="gray")
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.title(str(labels[i]), fontsize=6)
    plt.show()

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train_g = convert_to_gray(x_train)
x_test_g = convert_to_gray(x_test)

plot_sample(x_train, y_train)
plot_sample(x_train_g, y_train)
