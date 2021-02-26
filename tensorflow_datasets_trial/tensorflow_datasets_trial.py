#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

#datasets = tfds.load("mnist", data_dir="data/datasets")
#datasets = tfds.load("cifar10", data_dir="data/datasets")

#
# ImageNet2012
#
#  Before you start, download the following file into the following directory
#    ILSVRC2012_img_train.tar
#    ILSVRC2012_img_val.tar
#    -> data/datasets/downloads/manual
#

# # as dictionary
# datasets = tfds.load("imagenet2012", data_dir="data/datasets")
# 
# ds_train = datasets["train"]
# ds10 = ds_train.take(50)
# 
# for i, d in enumerate(ds10):
#     plt.subplot(5, 10, i+1)
#     plt.imshow(d["image"])
#     plt.title(str(d["label"].numpy()))
#     plt.xticks([]), plt.yticks([])
# plt.show()

# # as supervised -> return (image, label) pair
# datasets = tfds.load("imagenet2012", data_dir="data/datasets", as_supervised=True)
# 
# ds_train = datasets["train"]
# ds10 = ds_train.take(50)
# 
# for i, (image, label) in enumerate(ds10):
#     plt.subplot(5, 10, i+1)
#     plt.imshow(image)
#     plt.title(label.numpy())
#     plt.xticks([]), plt.yticks([])
# plt.show()

# # as numpy
# datasets = tfds.load("imagenet2012", data_dir="data/datasets", as_supervised=True)
# 
# ds_train = datasets["train"]
# ds10 = ds_train.take(50)
# 
# for i, (image, label) in enumerate(tfds.as_numpy(ds10)):
#     plt.subplot(5, 10, i+1)
#     plt.imshow(image)
#     plt.title(label)
#     plt.xticks([]), plt.yticks([])
# plt.show()


# datasets = tfds.load("imagenet2012", data_dir="data/datasets", as_supervised=True, batch_size=10)

# # visualize
# datasets, info = tfds.load("imagenet2012", data_dir="data/datasets", as_supervised=True, with_info=True)
# tfds.show_examples(datasets["train"], info)

# map to same size
def transform_img(image, label):
    image = tf.cast(image, tf.float32) / 255
    if tf.shape(image)[0] < tf.shape(image)[1]:
        target_size = (256, tf.shape(image)[1]*256//tf.shape(image)[0])
    else:
        target_size = (tf.shape(image)[0]*256//tf.shape(image)[1], 256)
    resized = tf.image.resize(image, target_size, method="bicubic")
    cropped = tf.image.random_crop(resized, (224, 224, 3))
    return cropped, label

(ds_train, ds_valid), ds_info = tfds.load("imagenet2012", data_dir="data/datasets",
                                          split=("train", "validation"),
                                          as_supervised=True, with_info=True)

num_train = ds_info.splits["train"].num_examples
num_valid = ds_info.splits["validation"].num_examples

#ds_train.map(transform_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#ds_train.cache()
#ds_train.batch(1)
#ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_train = ds_train.map(transform_img)

ds10 = ds_train.take(50)

for i, (image, label) in enumerate(ds10):
    plt.subplot(5, 10, i+1)
    plt.imshow(image)
    plt.title(label.numpy())
    plt.xticks([]), plt.yticks([])
plt.show()
