#!/usr/bin/env python

import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
assert(tf.__version__ == "2.15.0")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import timeit

class my_dense(tf.Module):
    def __init__(self, idim, odim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self.name_scope:
            self.w = tf.Variable(tf.random.normal((idim, odim), 0.0, 0.001),
                                 name="w")
            self.b = tf.Variable(tf.zeros((odim,)), name="b")

    @tf.function
    def __call__(self, idata):
        return tf.linalg.matmul(idata, self.w) + self.b

class my_module(tf.Module):
    def __init__(self, idim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self.name_scope:
            self.dense0 = my_dense(2, idim, name="dense0")
            self.dense1 = my_dense(idim, 1, name="dense1")

    # have to specify input batch size as None in order to
    # make the saved_model executable with any input batch size
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2],
                                                dtype=tf.float32)])
    def __call__(self, idata):
        d0 = self.dense0(idata)
        d0a = tf.nn.relu(d0)
        d1 = self.dense1(d0a)
        return tf.squeeze(d1)

def test_dense():
    dense = my_dense(2, 1)

    idata = tf.random.uniform((1000, 2), 0, 1)
    odata = tf.reduce_sum(idata, axis=1) + 0.5
    #idata += tf.random.normal(idata.shape, 0.0, 0.01)

    optimizer = tf.keras.optimizers.Adam()
    losser = tf.keras.losses.MeanSquaredError()
    batch_size = 100
    epochs = 500

    for i in range(epochs):
        sum_loss = 0
        for j in range(idata.shape[0]//batch_size):
            with tf.GradientTape() as tape:
                pred = dense(idata[j*batch_size:(j+1)*batch_size])
                #loss = tf.keras.losses.MSE(odata[j*batch_size:(j+1)*batch_size], pred)
                #loss = tf.reduce_mean(loss)
                loss = losser(odata[j*batch_size:(j+1)*batch_size], pred)
            gradients = tape.gradient(loss, dense.trainable_variables)
            optimizer.apply_gradients(zip(gradients, dense.trainable_variables))

            sum_loss += loss.numpy()
        print(i, sum_loss)

    print(dense.w)
    print(dense.b)

#test_dense()

start_time = timeit.default_timer()
skipped_time = 0

module = my_module(5)

idata = tf.random.uniform((1000, 2), -1, 1)
odata = tf.reduce_sum(idata**2, axis=1) + 0.5

optimizer = tf.keras.optimizers.Adam()
losser = tf.keras.losses.MeanSquaredError()
batch_size = 100
epochs = 500

loss_hist = []
for i in range(epochs):
    sum_loss = 0
    for j in range(idata.shape[0]//batch_size):
        with tf.GradientTape() as tape:
            pred = module(idata[j*batch_size:(j+1)*batch_size])
            loss = losser(odata[j*batch_size:(j+1)*batch_size], pred)
        gradients = tape.gradient(loss, module.trainable_variables)
        optimizer.apply_gradients(zip(gradients, module.trainable_variables))

        sum_loss += loss.numpy()
    print_time = timeit.default_timer()
    print(i, sum_loss)
    skipped_time += timeit.default_timer() - print_time
    loss_hist += [sum_loss]

# pred = module(idata)
# 
# ax = Axes3D(plt.figure())
# ax.plot(idata[:, 0], idata[:, 1], odata, ".", ms=4, mew=0.5, label="true")
# ax.plot(idata[:, 0], idata[:, 1], pred, ".", ms=4, mew=0.5, label="pred")
# ax.legend()
# 
# plt.figure()
# plt.plot(loss_hist)
# 
# plt.show()

print("Elapsed time: ", timeit.default_timer() - start_time - skipped_time)

tf.saved_model.save(module, "model")

imported = tf.saved_model.load("model")

pred = imported(idata)

ax = Axes3D(plt.figure())
ax.plot(idata[:, 0], idata[:, 1], odata, ".", ms=4, mew=0.5, label="true")
ax.plot(idata[:, 0], idata[:, 1], pred, ".", ms=4, mew=0.5, label="pred")
ax.legend()

plt.figure()
plt.plot(loss_hist)

plt.show()
