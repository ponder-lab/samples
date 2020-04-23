#!/usr/bin/env python

import concurrent.futures as futures
import random
import math

def func(id):
    print("%dth process" % id)

    sum = 0.0
    for i in range(10000000):
        sum += math.sqrt(math.exp(random.normalvariate(10.0, 1.0)))

    print("%dth processe's result = %f" % (id, sum))

    return sum

def main():
    # # Only executing
    # with futures.ProcessPoolExecutor(max_workers=6) as executor:
    #     for i in range(6):
    #         executor.submit(func, i)

    # # Getting return value
    # with futures.ProcessPoolExecutor(max_workers=6) as executor:
    #     res = []
    #     for i in range(6):
    #         res += [executor.submit(func, i)]
    # 
    # for r in res:
    #     print(r.result())

    # Using map (pass iterator and get generator)
    with futures.ProcessPoolExecutor(max_workers=6) as executor:
        res = executor.map(func, range(6))

    for r in res:
        print(r)

if __name__ == "__main__":
    main()


