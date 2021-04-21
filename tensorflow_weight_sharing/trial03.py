#!/usr/bin/env python

import numpy as np
import tensorflow as tf

@tf.function
def my_conv2d(inputs, w, b, kernel_size, strides, padding):

    # Extract patch
    #  patches dim: (batch, h, w, kernel_size[0]*kernel_size[1]*ich)
    patch_sizes = (1,) + kernel_size + (1,)
    patch_strides = (1,) + strides + (1,)
    patches = tf.image.extract_patches(inputs, sizes=patch_sizes,
                                       strides=patch_strides,
                                       rates=(1, 1, 1, 1),
                                       padding=padding)

    # Multiply weights
    mul =  w * tf.expand_dims(patches, 3)
    madd = tf.reduce_sum(mul, axis=-1)
    maddb = madd + b

    return maddb

class SharedConvs(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        ich = 32
        och = 32
        kernel_size = (3, 3)
        
        weights_shape = (och,
                         kernel_size[0] * kernel_size[1] * ich)
        w = tf.random.normal(weights_shape, stddev=0.01)
        self.w = self.add_weight(shape=w.shape, trainable=True)
        self.w.assign(w)

        b = tf.zeros((och))
        self.b = self.add_weight(shape=b.shape, trainable=True)
        self.b.assign(b)

    def call(self, inputs):
        x = inputs
        
        x = my_conv2d(x, self.w, self.b, (3, 3), (1, 1), "SAME")
        x = tf.nn.relu(x)
        x = my_conv2d(x, self.w, self.b, (3, 3), (1, 1), "SAME")
        x = tf.nn.relu(x)

        x = tf.nn.max_pool2d(x, (2, 2), (2, 2), "VALID")

        x = my_conv2d(x, self.w, self.b, (3, 3), (1, 1), "SAME")
        x = tf.nn.relu(x)
        x = my_conv2d(x, self.w, self.b, (3, 3), (1, 1), "SAME")
        x = tf.nn.relu(x)

        x = tf.nn.max_pool2d(x, (2, 2), (2, 2), "VALID")

        outputs = x
        return outputs
    
class MyModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)

        self.conv0 = tf.keras.layers.Conv2D(32, (3, 3), padding="SAME")
        self.act0 = tf.keras.layers.Activation("relu")

        self.shared_convs = SharedConvs()

        self.flat4 = tf.keras.layers.Flatten()
        self.dense5 = tf.keras.layers.Dense(10)
        self.act5 = tf.keras.layers.Activation("softmax")

    def call(self, inputs):
        x = inputs

        x = self.conv0(x)
        x = self.act0(x)

        x = self.shared_convs(x)

        x = self.flat4(x)
        x = self.dense5(x)
        x = self.act5(x)
        outputs = x
        return outputs

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32)[..., None] / 255

model = MyModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.build(x_train.shape)
model.summary()

model.fit(x_train, y_train,
          batch_size=100, epochs=10, verbose=1,
          validation_split=0.1)
