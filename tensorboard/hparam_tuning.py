#!/usr/bin/env python

import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

fname = os.path.splitext(__file__)[0]
dname_logs = "result_" + fname + "/logs"

# Load fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Set hyper parameters to tune
#HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_NUM_UNITS = hp.HParam('num_units', hp.IntInterval(16, 32))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer(dname_logs).as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

def train_test_model(hparams):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Run with 1 epoch to speed things up for demo purposes
    model.fit(x_train, y_train, epochs=1)
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0
#for num_units in HP_NUM_UNITS.domain.values:
#    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
#        for optimizer in HP_OPTIMIZER.domain.values:
#            hparams = {
#                HP_NUM_UNITS: num_units,
#                HP_DROPOUT: dropout_rate,
#                HP_OPTIMIZER: optimizer,
#            }
#            run_name = "run-%d" % session_num
#            print('--- Starting trial: %s' % run_name)
#            print({h.name: hparams[h] for h in hparams})
#            run(dname_logs + '/' + run_name, hparams)
#            session_num += 1
for i in range(20):
    num_units = HP_NUM_UNITS.domain.sample_uniform()
    dropout_rate = HP_DROPOUT.domain.sample_uniform()
    optimizer = HP_OPTIMIZER.domain.sample_uniform()
    hparams = {
        HP_NUM_UNITS: num_units,
        HP_DROPOUT: dropout_rate,
        HP_OPTIMIZER: optimizer,
    }
    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run(dname_logs + '/' + run_name, hparams)
    session_num += 1
