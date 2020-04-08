#!/usr/bin/env python

import random
import operator
import matplotlib.pyplot as plt

import cgp

# Define ground truth function
# (or provide input data and expected data)
def func_gt(idata):
    odata = [idata[0]**2 + idata[1]**2,
             idata[0]**2 - idata[1]**2]
    return odata

# Define fitness function (must be larger the better)
def func_fitness(exp, odata):
    sum = 0.0
    for e, o in zip(exp, odata):
        sum += (e - o)**2
    return -sum

# Set random seed if necessary
random.seed(1)

# Define network parameters
num_input = 2
num_output = 2
rows = 2
cols = 2
levels_back = 2

ops = cgp.operators([operator.add, operator.sub, operator.mul, operator.abs],
                    [2, 2, 2, 1])

# Define training parameters
num_genotype = 5
num_generations = 1000
num_mutate = 2

# Provide training data (input_data and expected_data)
idata_list = []
for i in range(-2, 3):
    for j in range(-2, 3):
        idata_list += [[i, j]]

expdata_list = []
for idata in idata_list:
    expdata_list += [func_gt(idata)]

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
    #print("Curr. Best genotype =", genotypes[parent_idx])

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

plt.plot(max_hist)
plt.savefig("test1_hist.png")
plt.show()

best_genotype.draw_graph(ops, "test1_graph")
