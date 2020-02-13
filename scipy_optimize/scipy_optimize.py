import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

def f(x):
    return x[0]**2 + x[1]**2

def fd(x):
    return np.array([2.0*x[0], 2.0*x[1]])

def main():
    #x = np.arange(-5, 5.1, 0.1)
    #plt.plot(x, f(x))
    #plt.show()

    x_init = np.array([1.0, 1.0], dtype='float32')
    res = optimize.fmin(f, x_init)
    print(res)
    res = optimize.fmin_powell(f, x_init)
    print(res)
    res = optimize.fmin_cg(f, x_init)
    print(res)
    res = optimize.fmin_bfgs(f, x_init, fprime=fd)
    print(res)
    res = optimize.fmin_ncg(f, x_init, fprime=fd)
    print(res)


if __name__ == "__main__":
    main()
