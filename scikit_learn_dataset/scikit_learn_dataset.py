#!/usr/bin/env python

import matplotlib.pyplot as plt

import numpy as np

# it doesnt work if import sklern, then use as sklearn.datasets.xxx
# instead import sklearn.datasets, then use as aklearn.datasets.xxx
import sklearn.datasets

def show_points(x):
    cmap = plt.get_cmap("tab10")

    num_cluster = np.max(x[1]) + 1
    for i in range(num_cluster):
        xc = x[0][x[1] == i]
        plt.scatter(xc[:, 0], xc[:, 1], c=cmap(i))
    plt.show()

x = sklearn.datasets.make_moons(1000, noise=0.05, random_state=0)
show_points(x)

x = sklearn.datasets.make_circles(1000, noise=0.05, random_state=0, factor=0.6)
show_points(x)

x = sklearn.datasets.make_blobs(1000, n_features=2, centers=3, cluster_std=0.5, random_state=0)
show_points(x)
