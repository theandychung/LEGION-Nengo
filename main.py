from vfield import VectorField
import matplotlib.pyplot as plt
from Constants import *
import numpy as np
import nengo
from nengo.utils.functions import piecewise
# plt.close('all')

def oscillator(num_neurons= 200, tau=9, label=None, net=None):
    if net is None:
        net = nengo.Network()
    with net:
        # net.input = nengo.Node(size_in= 1)
        net.ensemble = nengo.Ensemble(
            n_neurons= num_neurons, dimensions= 2, radius= 8, label=label)

        # osc to osc connection
        def feedback(x):
            x, y = x#, x_avg
            dx = 3 * x - x ** 3 + 2 - y + np.random.normal(0,1,None)
            dy = epsilon * (gamma * (1 + np.tanh(x / beta)) - y)
            return [tau * dx + x, tau * dy + y]#, tau*x+.01*x_avg

        nengo.Connection(net.ensemble, net.ensemble, function= feedback, synapse= tau)
        # inp to osc connection
        # nengo.Connection(net.input, net.ensemble[0], transform= tau, synapse= tau)
    return net

def inhibitor(num_neurons=150, tau=2.5, net=None):
    if net is None:
        net = nengo.Network()
    with net:
        net.input = nengo.Node(size_in=1, output=lambda t, x: 1 if x >= .9 else 0)#, output=lambda t, x: 1 if x >= .9 else 0
        net.ensemble = nengo.Ensemble(n_neurons=num_neurons,
                                      dimensions=1,
                                      radius=1)
        nengo.Connection(net.input, net.ensemble, transform= tau* phi, synapse=tau)
        nengo.Connection(net.ensemble, net.ensemble, transform= [-tau*phi+1], synapse=tau)
    return net

def s_f(x, theta):
    s = 1 / (1 + np.exp(-kappa * (x - theta)))
    return s


def T_f(xi_f,xi_t, xj_f, xj_t):
    W_permanent = np.exp(-(xj_t - xi_t)**2 / sigma_t**2 + (xj_f - xi_f)**2 / sigma_f**2)
    return W_permanent


def inhib2local_connect(Inh, Local, tau):
    #connection from inhib to local oscillators
    def weight(x):
        return -W1*s_f(x,theta_1)
    # def justfunction(x):
    #     return x*[-2.5 for i in range(num_neurons)]
    # nengo.Connection(Inh, Local.neurons, function=justfunction, synapse=tau)
    nengo.Connection(Inh, Local, function= weight, synapse = tau)


def local2inhib_connect(Local, Inh, tau):
    def justfun(x):
        if x >= theta_z:
            x = 1
        else:
            x = 0
        return x
    nengo.Connection(Local,Inh, function=justfun, synapse=tau)
    # nengo.Connection(Local, Inh, synapse=tau)

def local2local_connect(Local1, Local2, tau):
    nengo.Connection(Local1, Local2, transform= W0, synapse=tau)


# net = nengo.Network(label="withseed",seed=124)
net = nengo.Network(label="withoutseed")
with net:
    net.config[nengo.Ensemble].neuron_type = nengo.Direct()# force direct
    net.num = 1 #num doesn't matter when we are using nengo.Direct()
    net.tau = 3.0
    net.inhib = inhibitor(net.num, net.tau)


    grid_r, grid_c = inp.shape
    net.ea_oscillator = [[0 for x in range(grid_c)] for x in range(grid_r)]
    net.ea_collector = [0 for x in range(grid_r)]

    for i in range(grid_r):
        net.ea_collector[i] = nengo.Ensemble(net.num, dimensions= grid_c, radius=1)

    # create input nodes, oscillators, and
    # establish the connection between oscillators and inhib
    for i in range(grid_r):
        for j in range(grid_c):
            net.inputnode = nengo.Node(inp[i][j], label="input%d%d" % (i, j))
            net.ea_oscillator[i][j] = oscillator(net.num, net.tau)
            nengo.Connection(net.inputnode,
                             net.ea_oscillator[i][j].ensemble[0],
                             synapse = net.tau)
            inhib2local_connect(net.inhib.ensemble,
                                net.ea_oscillator[i][j].ensemble[0],
                                net.tau)
            local2inhib_connect(net.ea_oscillator[i][j].ensemble[0],
                                net.inhib.input,
                                net.tau)
            nengo.Connection(net.ea_oscillator[i][j].ensemble[0],
                             net.ea_collector[i][j],
                             net.tau)

    for i in range(grid_r):
        for j in range(grid_c):
            if i - 1 >= 0:
                local2local_connect(net.ea_oscillator[i][j].ensemble[0],
                                    net.ea_oscillator[i - 1][j].ensemble[0],
                                    net.tau)
            if i + 1 < grid_r:
                local2local_connect(net.ea_oscillator[i][j].ensemble[0],
                                    net.ea_oscillator[i + 1][j].ensemble[0],
                                    net.tau)
            if j - 1 >= 0:
                local2local_connect(net.ea_oscillator[i][j].ensemble[0],
                                    net.ea_oscillator[i][j - 1].ensemble[0],
                                    net.tau)
            if j + 1 < grid_c:
                local2local_connect(net.ea_oscillator[i][j].ensemble[0],
                                    net.ea_oscillator[i][j + 1].ensemble[0],
                                    net.tau)

#probes
    inhibitor_probe = nengo.Probe(net.inhib.ensemble, synapse= 0.01)

    # oscillator_probes = [0 for x in range(grid_r)]
    # for i in range(grid_r):
    #     oscillator_probes[i] = nengo.Probe(net.ea_collector[i], synapse=0.01)
    oscillator_probes = [[0 for x in range(grid_c)] for x in range(grid_r)]
    for i in range(grid_r):
        for j in range(grid_c):
            oscillator_probes[i][j] = nengo.Probe(net.ea_oscillator[i][j].ensemble[0],synapse=0.01)

with nengo.Simulator(net) as sim:
    sim.run(30)
    t = sim.trange()
    plt.figure()
    plt.subplot(grid_r+1,1,1)
    plt.plot(t, sim.data[inhibitor_probe], label= "inhibitor", color= 'k')
    plt.legend(prop={'size':13})
    for i in range(grid_r):
        plt.subplot(grid_r + 1, 1, i + 2)
        for j in range(grid_c):
            plt.plot(t, sim.data[oscillator_probes[i][j]],
                     label="oscillator %d" % (j+1))
    plt.legend(prop={'size': 13})
    plt.suptitle('LEGION')
    plt.ylabel('Activities')
    plt.xlabel('Time (sec)')
    plt.show()
