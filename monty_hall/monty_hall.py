#!/usr/bin/env python

import random

stay_sum = 0
change_sum = 0
for i in range(100):
    boxes = [0, 0, 0]

    win_idx = win_idx
    boxes[win_idx] = 1

    pick_idx = random.randint(0, 2)

    for j in range(len(boxes)):
        if j != win_idx and j != pick_idx:
            open_idx = j

    for j in range(len(boxes)):
        if j != pick_idx and j != open_idx:
            remain_idx = j

    stay_sum += boxes[pick_idx]
    change_sum += boxes[remain_idx]

print("Stay sum = %d" % stay_sum)
print("Change sum = %d" % change_sum)
