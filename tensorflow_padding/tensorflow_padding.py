#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

import my_layers

def plot_sample(imgs_g):
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(imgs_g[i])
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.show()

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype(np.float32) / 255.0

# Reflect
input_node = keras.layers.Input(shape=x_train.shape[1:])
pad = my_layers.Padding2D(5, "reflect")(input_node)
model = keras.models.Model(inputs=input_node, outputs=pad)
model.summary()

x_reflect = model.predict(x_train)
plot_sample(x_reflect)

# Symmetric
input_node = keras.layers.Input(shape=x_train.shape[1:])
pad = my_layers.Padding2D(5, "symmetric")(input_node)
model = keras.models.Model(inputs=input_node, outputs=pad)
model.summary()

x_symmetric = model.predict(x_train)
plot_sample(x_symmetric)

# Replicate
input_node = keras.layers.Input(shape=x_train.shape[1:])
pad = my_layers.Padding2D(1, "symmetric")(input_node)
for i in range(4):
    pad = my_layers.Padding2D(1, "symmetric")(pad)
model = keras.models.Model(inputs=input_node, outputs=pad)
model.summary()

x_replicate = model.predict(x_train)
plot_sample(x_replicate)
