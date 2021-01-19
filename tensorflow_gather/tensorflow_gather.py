#!/usr/bin/env python

import tensorflow as tf

#
# tf.gather()
#

# 1D data
data = tf.range(10)
print("\n**** tf.gather() usage ****")
print("\n1D data is", data, "\n")
print("tf.gather(data, 5) =", tf.gather(data, 5))
print("tf.gather(data, [3, 7, 5, 2]) =", tf.gather(data, [3, 7, 5, 2]))
print("tf.gather(data, [[1, 3, 5], [2, 8, 9]]) =", tf.gather(data, [[1, 3, 5], [2, 8, 9]]))

# 2D data
data = tf.reshape(tf.range(12), (3, 4))
print("\n2D data is", data, "\n")
print("tf.gather(data, 1) =", tf.gather(data, 1))
print("tf.gather(data, [0, 2]) =", tf.gather(data, [0, 2]))
print("tf.gather(data, 2, axis=1) =", tf.gather(data, 2, axis=1))
print("tf.gather(data, [1, 3], axis=1) =", tf.gather(data, [1, 3], axis=1))

#
# tf.gather_nd()
#

data = tf.reshape(tf.range(24), (2, 3, 4))
print("\n**** tf.gather_nd() usage ****")
print("\n3D data is", data, "\n")
print("tf.gather_nd(data, [0]) =", tf.gather_nd(data, [0]))
print("tf.gather_nd(data, [1, 1]) =", tf.gather_nd(data, [1, 1]))
print("tf.gather_nd(data, [1, 2, 3]) =", tf.gather_nd(data, [1, 2, 3]))
print("tf.gather_nd(data, [[1], [0]]) =", tf.gather_nd(data, [[0], [1]]))
print("tf.gather_nd(data, [[1, 1], [0, 2]]) =", tf.gather_nd(data, [[1, 1], [0, 2]]))
print("tf.gather_nd(data, [[1, 2, 3], [0, 1, 1]]) =", tf.gather_nd(data, [[1, 2, 3], [0, 1, 1]]))
print("tf.gather_nd(data, [[1], [0]], batch_dims=1) =", tf.gather_nd(data, [[1], [0]], batch_dims=1))
print("tf.gather_nd(data, [[1, 1], [0, 2]], batch_dims=1) =", tf.gather_nd(data, [[1, 1], [0, 2]], batch_dims=1))
