#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def preprocess_data(x_train, y_train, x_test, y_test, validation_split):

    # Add channel dimension if necessary
    if x_train.ndim == 3:
        x_train = x_train[..., None]
        x_test = x_test[..., None]

    # Convert to float and normalize
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # Random shuffle training data
    pidx = np.random.permutation(np.arange(x_train.shape[0]))
    x_train = x_train[pidx]
    y_train = y_train[pidx]

    # Divide into train and validation
    num_train = int(x_train.shape[0] * (1.0 - validation_split))
    x_valid = x_train[num_train:]
    x_train = x_train[:num_train]
    y_valid = y_train[num_train:]
    y_train = y_train[:num_train]

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def get_model(input_shape, num_label):

    input_node = tf.keras.layers.Input(input_shape)
    x = input_node

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(num_label, (3, 3), padding="same")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Activation("softmax")(x)

    output_node = x
    model = tf.keras.models.Model(input_node, output_node)
    return model

# MNIST:         60000 x 28 x 28 @uint8 -> 60000 @uint8 label[0-9]
# fashion MNIST: 60000 x 28 x 28 @uint8 -> 60000 @uint8 label[0-9]
# Cifar10:       50000 x 32 x 32 x 3 @uint8 -> 50000 x 1 @uint8 label[0-9]
# Cifar100:      50000 x 32 x 32 x 3 @uint8 -> 50000 x 1 @int64 label[0-99]
# Number of test: 10000(all datasets)

datasets = [tf.keras.datasets.mnist,
            tf.keras.datasets.fashion_mnist,
            tf.keras.datasets.cifar10,
            tf.keras.datasets.cifar100]
validation_split = 0.2
batch_size = 100
epochs = 100

fp = open("log.csv", "w")
print("dataset,train_loss,valid_loss,test_loss,train_acc,valid_acc,test_acc", file=fp)

for dataset in datasets:

    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train, y_train, \
        x_valid, y_valid, \
        x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test, validation_split)

    model = get_model(x_train.shape[1:], y_train.shape[1])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    dataset_name = dataset.__name__.split(".")[-1]
    fname_weights = dataset_name + "_weights.h5"
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=fname_weights,
                                                    save_best_only=True,
                                                    save_weights_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)]

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks, validation_data=(x_valid, y_valid))

    model.load_weights(fname_weights)
    score = model.evaluate(x_test, y_test, verbose=0)

    min_idx = np.argmin(np.array(history.history["val_loss"]))
    result1 = "%s,%f,%f,%f,%f,%f,%f" % (dataset_name,
                                        history.history["loss"][min_idx],
                                        history.history["val_loss"][min_idx],
                                        score[0],
                                        history.history["accuracy"][min_idx],
                                        history.history["val_accuracy"][min_idx],
                                        score[1])

    print(result1)
    print(result1, file=fp)

    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend(), plt.yscale("log"), plt.grid()
    plt.savefig(dataset_name + "_loss.png")
    plt.close()
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.legend(), plt.yscale("log"), plt.grid()
    plt.savefig(dataset_name + "_accuracy.png")
    plt.close()

fp.close()
