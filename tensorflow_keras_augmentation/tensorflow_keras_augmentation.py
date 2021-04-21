#!/usr/bin/env python

import sys
import time
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

def get_model(input_shape):
    input_node = keras.Input(input_shape)
    x = keras.layers.Convolution2D(64, (3, 3), activation="relu", padding="same")(input_node)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Convolution2D(64, (3, 3), activation="relu", padding="valid")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Convolution2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Convolution2D(64, (3, 3), activation="relu", padding="valid")(x)
    x = keras.layers.Flatten()(x)
    output_node = keras.layers.Dense(10, activation="softmax")(x)
    return keras.models.Model(input_node, output_node)

start = time.time()

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

(x_train_, y_train_), (x_test, y_test) = keras.datasets.mnist.load_data()

validation_split = 0.2
num_train = int(x_train_.shape[0] * (1.0 - validation_split))
num_valid = x_train_.shape[0] - num_train

x_train = (x_train_[:num_train].astype(np.float32) / 255.0)[..., tf.newaxis]
x_valid = (x_train_[num_train:].astype(np.float32) / 255.0)[..., tf.newaxis]
x_test = (x_test.astype(np.float32) / 255.0)[..., tf.newaxis]

y_train = y_train_[:num_train]
y_valid = y_train_[num_train:]

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1)

datagen.fit(x_train)

model = get_model(x_train[0].shape)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

num_epochs = 10
batch_size = 100
model_file = "tmp.h5"

callbacks = [keras.callbacks.ModelCheckpoint(filepath=model_file, \
                                             save_best_only=True)]

model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
          validation_data=(x_valid, y_valid),
          steps_per_epoch=len(x_train)/batch_size, epochs=num_epochs,
          callbacks=callbacks, verbose=1)

model = keras.models.load_model(model_file)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test score:", score[0])
print("Test accuracy:", score[1])

print("Elapsed time:", time.time() - start)
