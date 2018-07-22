from const import *
def s_f(x, theta):
    """
    This function sets as a filter for the input of inhibitor.
    Only strong input from oscillators can stimulate the inhibitor.
    The filter can be changed by the value kappa and theta defined in const.py
    :param x: input signal from a single oscillator.
    :param theta: Default 0.05.
    :return: stimulation value for the inhibitor.
    """
    s = 1 / (1 + np.exp(-kappa * (x - theta)))
    return s

def T_f(xi, xj):
    """
    This is the permanent weighting between two oscillators.
    The value of weighting changes depending not only on the pre-set value sigma_t and sigma_f, but also on the location of the oscillators on LEGION.
    Note that this function has been temporarily disabled and replaced by a constant value W0 in this project for better performance.
    :param xi: location of oscillator i (from 0 to inf)
    :param xj: location of oscillator j (from 0 to inf)
    :return: permanent weighting value
    """
    W_permanent = np.exp(-((xj.x_t - xi.x_t)**2 / sigma_t**2 + (xj.x_f - xi.x_f)**2 / sigma_f**2))
    return W_permanent

def convertinp(i):
    """
    This function ensures the input of each oscillator is capable of exciting the oscillator.
    If the input is larger than 0, input will be set to 0.2. Otherwise the input will be set to -0.02 to avoid noise.
    :param i: input of oscillator
    :return: 0.2 or -0.02
    """
    if i > 0:
        i = 0.2
    else:
        i = -0.02
    return i

