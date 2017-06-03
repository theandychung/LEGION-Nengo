import numpy as np
import matplotlib.pyplot as plt
import nengo
from nengo.processes import WhiteSignal
from nengo.utils.functions import piecewise

class Delay(object):
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((timesteps, dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        return self.history[0]


dt = 0.001
deltat=0.2
delay = Delay(1, timesteps=int(deltat / dt))
tau = 0.1
model = nengo.Network(label="Delayed connection")
with model:
    # We'll use white noise as input
    inp = nengo.Node(
        piecewise({
            0: 2,
            1: 4,
            2: 0,
            3: -2,
            4: 0,
            # 5: 0
        }))
    # inp = nengo.Node(lambda t: np.sin(8 * t))
    A = nengo.Ensemble(40, dimensions=1, neuron_type= nengo.Direct())
    nengo.Connection(inp, A, transform=tau, synapse=tau)
    nengo.Connection(A, A, transform=1, synapse=tau)

    delaynode = nengo.Node(delay.step, size_in=1, size_out=1)
    nengo.Connection(inp, delaynode)

    # Send the delayed output through an ensemble
    B = nengo.Ensemble(40, dimensions=1,neuron_type=nengo.Direct())
    nengo.Connection(delaynode, B, transform=-tau, synapse=tau)
    nengo.Connection(B, B, transform=1, synapse=tau)

    C = nengo.Ensemble(40, dimensions=1,neuron_type=nengo.Direct())
    def feedback(x):
        return x/(deltat)
    nengo.Connection(A, C, function=feedback, synapse=tau)
    nengo.Connection(B, C, function=feedback, synapse=tau)

    # Probe the input at the delayed output
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)
    C_probe = nengo.Probe(C, synapse=0.01)
    inp_probe = nengo.Probe(inp, synapse=0.01)
with nengo.Simulator(model) as sim:
    sim.run(6)
    # Plot the results
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(sim.trange(), sim.data[A_probe], lw=2, label="Integrated Input")
    plt.plot(sim.trange(), sim.data[B_probe], lw=2, label="Negative Integrated Input")
    plt.axvline(deltat, c='k')
    plt.tight_layout()
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(sim.trange(), sim.data[C_probe], lw=2, label="Differences")
    plt.plot(sim.trange(), sim.data[inp_probe], color='k', label="Original Input")
    plt.legend()
    plt.show()

