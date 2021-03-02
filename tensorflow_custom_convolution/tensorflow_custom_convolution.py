#!/usr/bin/env python

import time
import numpy as np
import tensorflow as tf

class MyConv2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding="VALID",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):

        weights_shape = (self.filters * self.kernel_size[0] * \
                         self.kernel_size[1] * input_shape[-1],)
        w = tf.random.normal(weights_shape, stddev=0.01)
        self.w = self.add_weight(shape=w.shape, trainable=True)
        self.w.assign(w)

        b = tf.zeros((self.filters))
        self.b = self.add_weight(shape=b.shape, trainable=True)
        self.b.assign(b)

        super().build(input_shape)
        
    def call(self, inputs):

        # Extract patch
        #  patches dim: (batch, h, w, kernel_size[0]*kernel_size[1]*ich)
        patch_sizes = (1,) + self.kernel_size + (1,)
        patch_strides = (1,) + self.strides + (1,)
        patches = tf.image.extract_patches(inputs, sizes=patch_sizes,
                                           strides=patch_strides,
                                           rates=(1, 1, 1, 1),
                                           padding=self.padding)

        # Multiply weights
        mul =  self.w * tf.expand_dims(patches, 3)
        madd = tf.reduce_sum(mul, axis=-1)
        maddb = madd + self.b

        return maddb

def get_model_normal(input_shape):
    input_node = tf.keras.Input(input_shape)
    x = input_node

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="valid")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="valid")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(10)(x)
    x = tf.keras.layers.Activation("softmax")(x)

    output_node = x
    return tf.keras.models.Model(input_node, output_node)

def get_model_myconv(input_shape):
    input_node = tf.keras.Input(input_shape)
    x = input_node

    x = tf.keras.layers.MyConv2D(32, (3, 3), padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.MyConv2D(32, (3, 3), padding="valid")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.MyConv2D(32, (3, 3), padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.MyConv2D(32, (3, 3), padding="valid")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(10)(x)
    x = tf.keras.layers.Activation("softmax")(x)

    output_node = x
    return tf.keras.models.Model(input_node, output_node)

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32)[..., None] / 255

model = get_model_normal(x_train[0].shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model(x_train[:2])

myconv_start = time.time()

model.fit(x_train, y_train,
          batch_size=100, epochs=10, verbose=1,
          validation_split=0.1)

myconv_end = time.time()

del model
model = get_model_normal(x_train[0].shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model(x_train[:2])

normal_start = time.time()

model.fit(x_train, y_train,
          batch_size=100, epochs=10, verbose=1,
          validation_split=0.1)

normal_end = time.time()

print("Normal convolution: %f[sec]" % (normal_end - normal_start))
print("My convolution: %f[sec]" % (myconv_end - myconv_start))
