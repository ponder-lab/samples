#!/usr/bin/env python

import codecs
import numpy as np
import tensorflow as tf

fname_alice = "11-0.txt"

with codecs.open(fname_alice, "r", encoding="utf-8") as fp:
    lines = [line.strip().lower() for line in fp if len(line.strip()) != 0]
    text = " ".join(lines)

chars = set(text)
nb_chars = len(chars)
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))

seqlen = 10
step = 1

input_chars = []
label_chars = []
for i in range(0, len(text) - seqlen, step):
    input_chars.append(text[i:i + seqlen])
    label_chars.append(text[i + seqlen])

X = np.zeros((len(input_chars), seqlen, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1

hidden_size = 128
batch_size = 128
num_iterations = 25
num_epochs_per_iteration = 1
num_preds_per_epoch = 100

model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(hidden_size, return_sequences=False,
                                    input_shape=(seqlen, nb_chars),
                                    unroll=True))
model.add(tf.keras.layers.Dense(nb_chars))
model.add(tf.keras.layers.Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

for iteration in range(num_iterations):
    print("*" * 50)
    print("Iteration #: %d" % (iteration))

    model.fit(X, y, batch_size=batch_size, epochs=num_epochs_per_iteration)

    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print("Generating from seed: %s" % (test_chars))
    print(test_chars, end="")
    for i in range(num_preds_per_epoch):
        Xtest = np.zeros((1, seqlen, nb_chars))
        for j, ch in enumerate(test_chars):
            Xtest[0, j, char2index[ch]] = 1

        pred = model.predict(Xtest, verbose=0)[0]
        ypred = index2char[np.argmax(pred)]
        print(ypred, end="")

        test_chars = test_chars[1:] + ypred
    print()

