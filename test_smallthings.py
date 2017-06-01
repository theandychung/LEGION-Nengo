import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.processes import WhiteSignal

class Delay(object):
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((timesteps, dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        return self.history[0]


dt = 0.001
delay = Delay(1, timesteps=int(0.2 / 0.001))
tau = 0.1
model = nengo.Network(label="Delayed connection")
with model:
    # We'll use white noise as input
    inp = nengo.Node(0.2, size_out=1)
    A = nengo.Ensemble(40, dimensions=1,neuron_type=nengo.Direct())
    nengo.Connection(inp, A)

    delaynode = nengo.Node(delay.step, size_in=1, size_out=1)
    nengo.Connection(inp, delaynode)

    # Send the delayed output through an ensemble
    B = nengo.Ensemble(40, dimensions=1,neuron_type=nengo.Direct())
    nengo.Connection(delaynode, B, transform=-1, synapse = tau)

    nengo.Connection(A, B, synapse= tau)
    def feedback(x):
        return x+tau*x/0.2
    nengo.Connection(B, B, synapse = tau)