import numpy as np
import tensorflow as tf

"""
Do an MNIST classification line by line by LSTM
"""

(x_train, y_train), \
    (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(None, 28)))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation("softmax"))
model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer="sgd",
              metrics=["accuracy"])

model.fit(x_train, y_train, validation_data=(x_test, y_test),
          batch_size=100, epochs=100)

