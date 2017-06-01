import numpy as np
import matplotlib.pyplot as plt
import nengo
import test_legion
model = nengo.Network(label='Ensemble Array')
with model:
    # Make an input node
    sin = nengo.Node(output=lambda t: [np.cos(t), np.sin(t)])

    # Make ensembles to connect
    A = nengo.networks.EnsembleArray(100, n_ensembles=2)
    B = nengo.Ensemble(100, dimensions=2)
    C = nengo.networks.EnsembleArrayy(100, n_ensembles=2)

    # Connect the model elements, just feedforward
    nengo.Connection(sin, A.input)
    nengo.Connection(A.output, B)
    nengo.Connection(B, C.input)

    # Setup the probes for plotting
    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A.output, synapse=0.02)
    B_probe = nengo.Probe(B, synapse=0.02)
    C_probe = nengo.Probe(C.output, synapse=0.02)
with nengo.Simulator(model) as sim:
    sim.run(10)
    plt.figure()
    plt.plot(sim.trange(), sim.data[sin_probe])
    plt.plot(sim.trange(), sim.data[A_probe])
    plt.plot(sim.trange(), sim.data[B_probe])
    plt.plot(sim.trange(), sim.data[C_probe]);
    plt.show()