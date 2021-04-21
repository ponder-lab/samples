#!/usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    def call(self, inputs):
        return inputs
        
class MyConv2D(tf.keras.layers.Layer):

    def __init__(self, shared_weights,
                 filters, kernel_size, strides=(1, 1), padding="VALID",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shared_weights = shared_weights
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):

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
        mul =  self.shared_weights.w * tf.expand_dims(patches, 3)
        madd = tf.reduce_sum(mul, axis=-1)
        maddb = madd + self.shared_weights.b

        return maddb

def get_model_myconv(input_shape):

    shared_weights = SharedWeights(32, (3, 3))
    
    input_node = tf.keras.Input(input_shape)
    x = input_node

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="SAME")(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = shared_weights(x)

    x = MyConv2D(shared_weights, 32, (3, 3), padding="SAME")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = MyConv2D(shared_weights, 32, (3, 3), padding="SAME")(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = MyConv2D(shared_weights, 32, (3, 3), padding="SAME")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = MyConv2D(shared_weights, 32, (3, 3), padding="SAME")(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(10)(x)
    x = tf.keras.layers.Activation("softmax")(x)

    output_node = x
    return tf.keras.models.Model(input_node, output_node)

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32)[..., None] / 255

model = get_model_myconv(x_train[0].shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model(x_train[:2])

model.fit(x_train, y_train,
          batch_size=100, epochs=10, verbose=1,
          validation_split=0.1)
