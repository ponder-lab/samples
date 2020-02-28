#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

def plot_sample(clean, noisy, denoised):
    plt.figure(figsize=(10, 3))
    for i in range(10):
        plt.subplot(3, 10, i+1)
        plt.imshow(clean[i])
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.subplot(3, 10, i+11)
        plt.imshow(noisy[i])
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.subplot(3, 10, i+21)
        plt.imshow(denoised[i])
        plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.show()
    plt.savefig("denoise_simple_ae.png")

def add_noise(data, std):
    noisy = data + np.random.normal(np.zeros(data.shape[-1], dtype=data.dtype), \
                                    np.ones(data.shape[-1], dtype=data.dtype) * std, \
                                    data.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy

def ae_model(input_shape):
    input_node = keras.layers.Input(shape=input_shape)

    num_ch = 64
    
    # Encoder
    conv0 = keras.layers.Conv2D(num_ch, (3, 3), activation="relu", padding="same")(input_node)
    pool0 = keras.layers.MaxPooling2D((2, 2), padding="same")(conv0)
    conv1 = keras.layers.Conv2D(num_ch, (3, 3), activation="relu", padding="same")(pool0)
    encoded = keras.layers.MaxPooling2D((2, 2), padding="same")(conv1)

    # Decoder
    conv2 = keras.layers.Conv2D(num_ch, (3, 3), activation="relu", padding="same")(encoded)
    uscl2 = keras.layers.UpSampling2D((2, 2))(conv2)
    conv3 = keras.layers.Conv2D(num_ch, (3, 3), activation="relu", padding="same")(uscl2)
    uscl3 = keras.layers.UpSampling2D((2, 2))(conv3)
    #decoded = keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(uscl3)
    decoded = keras.layers.Conv2D(3, (3, 3), padding="same")(uscl3)

    model = keras.models.Model(input_node, decoded)
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    
    return model

(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train_noisy = add_noise(x_train, 0.1)
x_test_noisy = add_noise(x_test, 0.1)

model = ae_model(x_train.shape[1:])
model.fit(x_train_noisy, x_train, \
          epochs=10, batch_size=128, validation_split=0.2)
model.save("denoise_simple_ae.h5")

# model = keras.models.load_model("denoise_simple_ae.h5")

x_decoded = model.predict(x_test_noisy)
x_decoded = np.clip(x_decoded, 0.0, 1.0)

plot_sample(x_test, x_test_noisy, x_decoded)
