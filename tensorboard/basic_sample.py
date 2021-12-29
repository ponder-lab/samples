#!/usr/bin/env python

import os
import tensorflow as tf
import datetime

fname = os.path.splitext(__file__)[0]

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train[..., None] / 255.0, x_test[..., None] / 255.0

def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, (3, 3), padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, (3, 3), padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

model = create_model()
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "result_" + fname + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()

def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 1e-2
    if epoch > 10:
        learning_rate = 1e-3

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=20, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback, lr_callback])

file_writer.flush()
