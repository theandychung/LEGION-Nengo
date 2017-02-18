import matplotlib.pyplot as plt
import nengo
from nengo.utils.functions import piecewise
import math

import numpy as np



#initial values
I = 0.11
gamma = 2.1
beta = 0.1
eta = 0.35

tau=.1;
model = nengo.Network(label='WT Oscillator')
with model:
    inp = nengo.Node(I)
    oscillator = nengo.Ensemble(50, dimensions=2, radius=10)
    nengo.Connection(inp,oscillator[0])
     # osc to osc connection
    def feedback(x):
        x,y = x
        dx =  3 * x - x**3 + 2 - y
        dy = eta * (gamma * (1 + math.tanh(x / beta)) - y)
        return [tau*dx+x, tau*dy+y]
    nengo.Connection(oscillator, oscillator, function=feedback, synapse=tau)
    # inp to osc connection
    nengo.Connection(inp,oscillator[0], transform = tau)



    stim_pr = nengo.Probe(x)

with nengo.Simulator(model) as sim:
    sim.run(20)

import matplotlib.pyplot as plt
plt.plot(sim.trange(),sim.data[stim_pr], label="stim output")
plt.legend();
plt.show()