#!/usr/bin/env python

import os
import tensorflow as tf
import datetime

dname_result = "result_" + os.path.splitext(__file__)[0]

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

cp_path = os.path.join(dname_result, "cp-{epoch:04d}")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=cp_path,
    verbose=1,
    save_weights_only=False,
    save_freq="epoch")

history = model.fit(x=x_train,
                    y=y_train,
                    epochs=10,
                    validation_data=(x_test, y_test),
                    callbacks=[cp_callback])

#model.save(os.path.join(dname_result, "saved"))

model2 = tf.keras.models.load_model(os.path.join(dname_result, "cp-0005"))
#model2 = tf.keras.models.load_model(os.path.join(dname_result, "saved"))

history2 = model2.fit(x=x_train,
                      y=y_train,
                      epochs=5,
                      validation_data=(x_test, y_test))

print("Original model:", model.evaluate(x_test, y_test))
print("Retrained model:", model2.evaluate(x_test, y_test))

import matplotlib.pyplot as plt
import numpy as np
plt.subplot(221)
plt.plot(history.history["loss"])
plt.plot(np.arange(5, 10), history2.history["loss"])
plt.subplot(222)
plt.plot(history.history["accuracy"])
plt.plot(np.arange(5, 10), history2.history["accuracy"])
plt.subplot(223)
plt.plot(history.history["val_loss"])
plt.plot(np.arange(5, 10), history2.history["val_loss"])
plt.subplot(224)
plt.plot(history.history["val_accuracy"])
plt.plot(np.arange(5, 10), history2.history["val_accuracy"])
plt.show()

