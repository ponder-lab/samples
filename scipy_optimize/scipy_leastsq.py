import numpy as np
from scipy import optimize

def f(x):
    return x**2 - 1

def main():
    res = optimize.leastsq(f, 10.0)
    print(res)

if __name__ == "__main__":
    main()
