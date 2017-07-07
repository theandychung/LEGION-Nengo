from const import *


def s_f(x, theta):
    s = 1 / (1 + np.exp(-kappa * (x - theta)))
    return s

def T_f(xi, xj):
    W_permanent = np.exp(-((xj.x_t - xi.x_t)**2 / sigma_t**2 + (xj.x_f - xi.x_f)**2 / sigma_f**2))
    return W_permanent

def convertinp(i):
    if i > 0:
        i = 0.2
    else:
        i = -0.02
    return i

