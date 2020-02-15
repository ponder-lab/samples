#!/usr/bin/env python

import time
import numpy as np
from numba import jit, f4, prange

#@jit
@jit(f4[:, :](f4[:], f4[:]), nopython=True)
#@jit(f4[:, :](f4[:], f4[:]), nopython=True, parallel=True)
def func(a, b):
    e = np.empty((len(a), len(b)), dtype=a.dtype)
    for i in prange(len(a)):
    #for i in range(len(a)):
        vala = a[i]
        for j in range(len(b)):
            e[i, j] = vala * b[j]
    return e

size = 1000
dtype = np.float32

a = np.array([np.arange(size)], dtype=dtype).T
b = np.array([np.arange(size)], dtype=dtype)

start = time.time()

c = np.matmul(a, b)

np_end = time.time()

d = np.empty((size, size), dtype=dtype)

for i, vali in enumerate(a):
    for j, valj in enumerate(a):
        d[i, j] = valj * vali

for_end = time.time()

e = func(a.T[0], b[0])

jit_end = time.time()

print("np time", np_end - start)
print("for time", for_end - np_end)
print("jit time", jit_end - for_end)
