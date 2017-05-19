import nengo
import matplotlib.pyplot as plt
import numpy as np
import Constants
from nengo.utils.functions import piecewise

model = nengo.Network()
with model:
    piecewise_f = piecewise({0: 1,
                             5: 0,
                             10:.8,
                             15:0,
                             20:1})
    input = nengo.Node(piecewise_f)
    ens = nengo.Ensemble(1,dimensions= 1, radius= 1, neuron_type= nengo.Direct())
    output = nengo.Ensemble(1,dimensions= 1, radius= 1, neuron_type= nengo.Direct())

    def s_f(x):
        s = 1/np.exp(-50 * (x - 10))
        return s

    nengo.Connection(input, ens, function=s_f, synapse= 0.9)
    nengo.Connection(ens, output, synapse= 0.9)

    input_pr = nengo.Probe(input, synapse=0.05)
    ens_pr = nengo.Probe(ens, synapse= 0.05)
    output_pr = nengo.Probe(output, synapse= 0.05)

with nengo.Simulator(model) as sim:
    sim.run(20)
    t = sim.trange()
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t, sim.data[input_pr], label='input')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(t, sim.data[ens_pr], label='ens')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(t, sim.data[output_pr], label= 'output')
    plt.legend()
    plt.show()