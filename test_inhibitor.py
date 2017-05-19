import nengo
import matplotlib.pyplot as plt
import Constants
from nengo.utils.functions import piecewise

def inhibitor(num_neurons=150, tau=2.5, net=None):
    if net is None:
        net = nengo.Network()
    with net:
        net.input = nengo.Node(size_in=1, output=lambda t, x: 1 if x >= .9 else 0)
        net.ensemble = nengo.Ensemble(n_neurons=num_neurons,
                                      dimensions=1,
                                      radius=1)
        nengo.Connection(net.input, net.ensemble, transform= tau* Constants.phi, synapse=tau)
        nengo.Connection(net.ensemble, net.ensemble, transform= [-Constants.phi+1], synapse=tau)
    return net


net = nengo.Network(label="inhibitor")
with net:
    net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    num = 40
    tau = .9
    T = 30
    net.inhib = inhibitor(num, tau)
    piecewise_f = piecewise({0:1,
                             5:0,
                             10:.5,
                             15:0,
                             20:3,
                             25:0})
    net.piecewise_inp = nengo.Node(piecewise_f)
    nengo.Connection(net.piecewise_inp, net.inhib.input)
    piecewise_inp_probe = nengo.Probe(net.piecewise_inp,synapse=0.01)
    input_probe = nengo.Probe(net.inhib.input, synapse=0.01)
    z_pr = nengo.Probe(net.inhib.ensemble, synapse=0.01)
with nengo.Simulator(net) as sim:
    sim.run(T)
    # <editor-fold desc="...plot">
    t = sim.trange()
    # z activities
    plt.figure()
    plt.plot(t, sim.data[piecewise_inp_probe], label="piecewise_inp", color='r')
    plt.plot(t, sim.data[input_probe], label="input", color='k')
    plt.plot(t, sim.data[z_pr], label="z activity", color='b')
    plt.title("Activities of the Oscillator")
    plt.xlabel("Time (s)")
    plt.ylabel("z activity")
    plt.ylim(-3, 7)
    plt.legend()
    # plt.grid()
    plt.show()
    # prettify()
    # plt.legend(['$x_0$', '$x_1$', '$\omega$']);
    # </editor-fold>