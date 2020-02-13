#!/usr/bin/env python

import numpy as np

a = np.array([[1, 2], [3, 4]], dtype=np.float32)
ab = a.tobytes()

hdr = np.array([a.ndim, a.shape[0], a.shape[1]], dtype=np.int32)
hdrb = hdr.tobytes()

with open("idata.bin", "wb") as f:
    f.write(hdrb)
    f.write(ab)

print("Original data =")
print(a)
