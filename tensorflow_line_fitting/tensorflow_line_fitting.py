#!/usr/bin/env python

import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

# y = x + 1
x = np.random.uniform(0.0, 1.0, 100).astype(np.float32)
y = (x + np.random.normal(0.0, 0.1, 100) + 1.0).astype(np.float32)

# y = ax + b
a = tf.Variable(np.random.normal(0.0, 0.1))
b = tf.Variable(np.random.normal(0.0, 0.1))

#optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for i in range(300):
    with tf.GradientTape() as tape:
        oval = a * tf.constant(x) + b
        loss = tf.keras.losses.MSE(tf.constant(y), oval)

    grad = tape.gradient(loss, [a, b])

    print("i=%d loss=%f a=%f b=%f" % (i, loss.numpy(), a.numpy(), b.numpy()))

    #rate = 0.1
    #a.assign(a - grad[0] * rate)
    #b.assign(b - grad[1] * rate)

    optimizer.apply_gradients(zip(grad, [a, b]))

xp = np.linspace(0.0, 1.0, 100)
plt.scatter(x, y)
plt.plot(xp, a.numpy() * xp + b.numpy(), color="red")
plt.show()

