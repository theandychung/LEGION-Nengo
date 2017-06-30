from Constants import *


class Movingavg(object):
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((timesteps, dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        if np.mean(self.history)>theta:
            temp = 1
        else:
            temp = 0
        return temp