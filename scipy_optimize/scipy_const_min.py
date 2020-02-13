import numpy as np
from scipy import optimize

def f(x):
    return np.linalg.norm(x, ord=1)

def df_each(x):
    return 1 if x >= 0 else -1

def df(x):
    return (df_each(x[0]), df_each(x[1]))

def eqcon(x):
    return x[0]**2 + x[1]**2 - 1

def main():
    res = optimize.fmin_slsqp(f, [11.0, 10.0], fprime=df, eqcons=[eqcon])
    print(res)

if __name__ == "__main__":
    main()
