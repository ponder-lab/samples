#!/usr/bin/env python

import numpy as np
import tensorflow as tf

class SharedWeights(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):

        weights_shape = (self.filters,
                         self.kernel_size[0] * self.kernel_size[1] * input_shape[-1])
        w = tf.random.normal(weights_shape, stddev=0.01)
        self.w = self.add_weight(shape=w.shape, trainable=True)
        self.w.assign(w)

        b = tf.zeros((self.filters))
        self.b = self.add_weight(shape=b.shape, trainable=True)
        self.b.assign(b)

        super().build(input_shape)

    def call(self, input_data):
        return [self.w, self.b]

class Thru(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, input_data):
        return input_data * 1

class MyConv2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding="VALID",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):

        super().build(input_shape)
        
    def call(self, input_data):

        inputs, w, b = input_data

        # Extract patch
        #  patches dim: (batch, h, w, kernel_size[0]*kernel_size[1]*ich)
        patch_sizes = (1,) + self.kernel_size + (1,)
        patch_strides = (1,) + self.strides + (1,)
        patches = tf.image.extract_patches(inputs, sizes=patch_sizes,
                                           strides=patch_strides,
                                           rates=(1, 1, 1, 1),
                                           padding=self.padding)

        # Multiply weights
        mul =  w * tf.expand_dims(patches, 3)
        madd = tf.reduce_sum(mul, axis=-1)
        maddb = madd + b

        return maddb

class MyModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)

        self.run_eagerly = True

        self.my_weights = SharedWeights(32, (3, 3))

        self.conv0 = tf.keras.layers.Conv2D(32, (3, 3), padding="SAME")
        self.act0 = tf.keras.layers.Activation("relu")

        self.thru1 = Thru()
        self.conv1 = MyConv2D(32, (3, 3), padding="SAME")
        self.act1 = tf.keras.layers.Activation("relu")
        self.thru2 = Thru()
        self.conv2 = MyConv2D(32, (3, 3), padding="SAME")
        self.act2 = tf.keras.layers.Activation("relu")

        self.pool2 = tf.keras.layers.MaxPooling2D()

        self.thru3 = Thru()
        self.conv3 = MyConv2D(32, (3, 3), padding="SAME")
        self.act3 = tf.keras.layers.Activation("relu")
        self.thru4 = Thru()
        self.conv4 = MyConv2D(32, (3, 3), padding="SAME")
        self.act4 = tf.keras.layers.Activation("relu")

        self.pool4 = tf.keras.layers.MaxPooling2D()
        self.flat4 = tf.keras.layers.Flatten()

        self.dense5 = tf.keras.layers.Dense(10)
        self.act5 = tf.keras.layers.Activation("softmax")

    def call(self, inputs):
        x = inputs

        x = self.conv0(x)
        x = self.act0(x)

        weights = self.my_weights(x)

        weights1 = self.thru1(weights)
        x = self.conv1([x,] + weights1)
        x = self.act1(x)

        weights2= self.thru2(weights)
        x = self.conv2([x,] + weights2)
        x = self.act2(x)
        x = self.pool2(x)

        weights3 = self.thru3(weights)
        x = self.conv3([x,] + weights3)
        x = self.act3(x)

        weights4 = self.thru4(weights)
        x = self.conv4([x,] + weights4)
        x = self.act4(x)

        x = self.pool4(x)
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
