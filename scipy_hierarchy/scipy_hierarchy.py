#!/usr/bin/env python

import numpy as np
import scipy.cluster.hierarchy as hierarchy
import matplotlib.pyplot as plt

def angle_between(a, b):
    cos = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    return np.rad2deg(np.arccos(np.abs(cos)))

# Random 2D vectors
a = np.random.random((64, 2))

# Hierachically cluster vectors by angles between each vector
#z = hierarchy.linkage(a, "inconsistent", angle_between) # Inconsistency (new_distance - mean) / std
#z = hierarchy.linkage(a, "average", angle_between) # Average distance
z = hierarchy.linkage(a, "complete", angle_between) # Max distance

# Plot hierarchy chart
#plt.figure(0)
#dns = hierarchy.dendrogram(z)
#plt.show()

# Cluster ID assigned for each data
#   cluster upto the distance between the clusters is less than t deg.
#t = hierarchy.fcluster(z, t=2.0, criterion="distance")
t = hierarchy.fcluster(z, t=5.0, criterion="distance")

# l: Root cluster of each cluster
# m: List of cluster IDs
l, m = hierarchy.leaders(z, t)

# Number of data in each cluster
num_member = np.empty(0, np.int64)
for i in range(len(l)):
    if l[i] < 64:
        num_member = np.append(num_member, 1)
    else:
        num_member = np.append(num_member, z[l[i] - 64, 3])
