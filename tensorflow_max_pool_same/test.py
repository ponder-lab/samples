#!/usr/bin/env python
"""
Find out what happens if multiple inputs into max_pooling have the same value.
"""
import tensorflow as tf

x = tf.Variable(tf.constant([[[0], [1], [1], [0]]], dtype=tf.float32))

with tf.GradientTape() as tape:
    y = tf.nn.max_pool1d(x, [1, 4, 1], [1, 1, 1], "VALID")
grad = tape.gradient(y, [x])

print(f"x={x}")
print(f"y={y}")
print(f"grad={grad}")
