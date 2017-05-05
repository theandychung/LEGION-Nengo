import nengo
import matplotlib.pyplot as plt
import Constants
from nengo.utils.functions import piecewise

def Inhibitor(num_neurons=150, tau=2.5, net=None):
    if net is None:
        net = nengo.Network()
    with net:
        net.input = nengo.Node(size_in=1)
        net.ensemble = nengo.Ensemble(n_neurons=num_neurons,
                                      dimensions=1,
                                      radius=1)
        nengo.Connection(net.input, net.ensemble, transform= [tau * Constants.phi], synapse=tau)
        nengo.Connection(net.ensemble, net.ensemble, transform=[-tau * Constants.phi + 1], synapse=tau)
    return net

def testinhib(net, inhib, T=10, dt=0.001):
    with net:
        piecewise_f = piecewise({0:1,
                                 5:0,
                                 10:5,
                                 15:0,
                                 20:-1,
                                 25:0})
        piecewise_inp = nengo.Node(piecewise_f)
        nengo.Connection(piecewise_inp, inhib.input)
        input_probe = nengo.Probe(inhib.input, synapse=0.01)
        z_pr = nengo.Probe(inhib.ensemble, synapse=0.01)
    with nengo.Simulator(net,dt=dt) as sim:
        sim.run(T)
    # <editor-fold desc="...plot">
    t = sim.trange()
    # z activities
    plt.figure()
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

net = nengo.Network(label="Inhibitor")
with net:
    net.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    num = 40
    net.inhib = Inhibitor(num)
    testinhib(net,net.inhib,40)