#!/usr/bin/env python

import random
import operator
import numpy as np

import matplotlib.pyplot as plt
import graphviz

import tensorflow.keras as keras
import cv2

import cgp

# Define ground truth function
# (or provide input data and expected data)

# Define fitness function (must be larger the better)
def func_fitness(exp, odata):
    sum = 0.0
    for e, o in zip(exp, odata):
        sum += np.sum((e - o)**2)
    return -sum

# Set random seed if necessary
np.random.seed(1)
random.seed(1)

# 2x2
# Define network parameters
num_input = 1
num_output = 1
rows = 2
cols = 2
levels_back = 2
# Define training parameters
num_genotype = 5
num_generations = 1000
num_mutate = 2
test_name = "test21"

"""
# 4x4
# Define network parameters
num_input = 1
num_output = 1
rows = 4
cols = 4
levels_back = 2
# Define training parameters
num_genotype = 5
num_generations = 1000
num_mutate = 2
test_name = "test22"
"""

"""
# 8x8
# Define network parameters
num_input = 1
num_output = 1
rows = 8
cols = 8
levels_back = 2
# Define training parameters
num_genotype = 5
num_generations = 10000
num_mutate = 2
test_name = "test23"
"""

"""
# 32 layer
# Define network parameters
num_input = 1
num_output = 1
rows = 2
cols = 32
levels_back = 2
# Define training parameters
num_genotype = 5
num_generations = 10000
num_mutate = 4
test_name = "test24"
"""

# Define operators

def blur(idata):
    return cv2.blur(idata, (3, 3))

def dilate(idata):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(idata, kernel)

def erode(idata):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(idata, kernel)

def laplacian(idata):
    return cv2.Laplacian(idata, cv2.CV_64F)

def sobel_x(idata):
    return cv2.Sobel(idata, cv2.CV_64F, 1, 0)

def sobel_y(idata):
    return cv2.Sobel(idata, cv2.CV_64F, 0, 1)

def normalize(idata):
    return idata / np.max(idata - np.min(idata))

ops = cgp.operators([np.add, np.subtract, np.multiply, np.abs, blur, dilate, erode, laplacian, sobel_x, sobel_y, normalize],
                    [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])

# Provide training data (input_data and expected_data)
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
#(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

idata_list = x_train[:30].astype(np.float64) / 255.0
idata_list = idata_list + np.random.normal(0.0, 0.1, idata_list.shape)
idata_list[np.where(idata_list > 1.0)] = 1.0
idata_list[np.where(idata_list < 0.0)] = 0.0

expdata_list = x_train[:30].astype(np.float64) / 255.0

idata_list = [[idata_list[i]] for i in range(idata_list.shape[0])]
expdata_list = [[expdata_list[i]] for i in range(expdata_list.shape[0])]

# Generate initial genotypes
genotypes = []
for i in range(num_genotype):
    genotypes += [cgp.genotype(num_input, num_output, rows, cols,
                               levels_back, ops)]

# Loop for each generation
max_hist = []
max_fitness = -1.0e10
for i in range(num_generations):
    fitness, parent_idx = cgp.find_best_genotype(genotypes, idata_list,
                                                 expdata_list, ops, func_fitness)
    print("[%d/%d]Curr. Max. fitness =" % (i, num_generations), fitness)
    # print("Curr. Best genotype =", genotypes[parent_idx])

    if fitness >= max_fitness:
        max_fitness = fitness
        best_genotype = genotypes[parent_idx].copy()

    max_hist += [max_fitness]

    genotypes[0] = genotypes[parent_idx].copy()
    
    for j in range(1, num_genotype):
        genotypes[j] = genotypes[parent_idx].copy()
        genotypes[j].mutate(num_mutate, ops)

print("Best genotype =", best_genotype)
print("Fitness =", max_fitness)

plt.figure(1)
for j in range(5):
    for i in range(6):
        plt.subplot(5, 6, j*6+i+1)
        plt.imshow(idata_list[j*6+i][0])
plt.savefig(test_name + "_idata.png")
plt.figure(2)
for j in range(5):
    for i in range(6):
        plt.subplot(5, 6, j*6+i+1)
        plt.imshow(expdata_list[j*6+i][0])
plt.savefig(test_name + "_expdata.png")
plt.figure(3)
for j in range(5):
    for i in range(6):
        plt.subplot(5, 6, j*6+i+1)
        plt.imshow(best_genotype.decode(idata_list[j*6+i], ops)[0])
plt.savefig(test_name + "_odata.png")
plt.figure(4)
plt.plot(max_hist)
plt.savefig(test_name + "_hist.png")
plt.show()

best_genotype.draw_graph(ops, test_name + "_graph")


