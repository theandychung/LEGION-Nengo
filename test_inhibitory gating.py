import numpy as np
import matplotlib.pyplot as plt

import nengo


n_neurons = 30

model = nengo.Network(label="Inhibitory Gating")
with model:
    A = nengo.Ensemble(n_neurons, dimensions=1)
    B = nengo.Ensemble(n_neurons, dimensions=1)
    C = nengo.Ensemble(n_neurons, dimensions=1)

from nengo.utils.functions import piecewise

with model:
    sin = nengo.Node(np.sin)
    inhib = nengo.Node(piecewise({0: 0, 2.5: 1, 5: 0, 7.5: 1, 10: 0, 12.5: 1}))
    nengo.Connection(sin, A)
    nengo.Connection(sin, B)
    nengo.Connection(inhib, A, transform=[[-2.5]])
    nengo.Connection(inhib, C)
    nengo.Connection(C, B, transform=[[-2.5]])

    sin_probe = nengo.Probe(sin)
    inhib_probe = nengo.Probe(inhib)
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)
    C_probe = nengo.Probe(C, synapse=0.01)

with nengo.Simulator(model) as sim:
    # Run it for 15 seconds
    sim.run(15)

    plt.figure()
    plt.plot(sim.trange(), sim.data[A_probe], label='Decoded output')
    plt.plot(sim.trange(), sim.data[sin_probe], label='Sine input')
    plt.plot(sim.trange(), sim.data[inhib_probe], label='Inhibitory signal')
    plt.legend()

    plt.figure()
    plt.plot(sim.trange(), sim.data[B_probe], label='Decoded output of B')
    plt.plot(sim.trange(), sim.data[sin_probe], label='Sine input')
    plt.plot(sim.trange(), sim.data[C_probe], label='Decoded output of C')
    plt.plot(sim.trange(), sim.data[inhib_probe], label='Inhibitory signal')
    plt.legend()

    plt.show()