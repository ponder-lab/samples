#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def train_step_generator():

    @tf.function
    def train_step(w, b, x, t, optimizer):
        with tf.GradientTape() as tape:
            y = tf.linalg.matmul(x, w) + b
            loss = tf.reduce_mean((y - t)**2)

        gradients = tape.gradient(loss, [w, b])

        #w.assign_sub(0.01 * gradients[0])
        #b.assign_sub(0.01 * gradients[1])
        optimizer.apply_gradients(zip(gradients, [w, b]))

        return loss

    return train_step


w_gt = np.array([1.0, 1.0], dtype=np.float32)
b_gt = np.array([1.0], dtype=np.float32)

x = np.random.uniform(-1.0, 1.0, (10000, 2)).astype(np.float32)
t = np.matmul(x, w_gt.reshape(2, 1)) + b_gt
t += np.random.normal(0, 1, t.shape)

loss_list = []
for j in range(2):
    w = np.random.normal(0, 0.1, (2, 1)).astype(np.float32)
    w = tf.Variable(w)
    b = np.random.normal(0, 0.1)
    b = tf.Variable(b)

    x = tf.constant(x)
    t = tf.constant(t)

    optimizer = tf.keras.optimizers.Adam(lr=0.02)

    train_step_fn = train_step_generator()

    loss_hist = np.empty(0, dtype=np.float32)
    for i in range(1000):
        loss = train_step_fn(w, b, x, t, optimizer)
        loss_hist = np.append(loss_hist, loss.numpy())

    loss_list += [loss_hist]

plt.plot(loss_list[0])
plt.plot(loss_list[1])
plt.grid()
plt.show()
