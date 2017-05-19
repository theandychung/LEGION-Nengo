


from vfield import VectorField
import matplotlib.pyplot as plt
import Constants
import numpy as np
import nengo
from nengo.utils.functions import piecewise
# plt.close('all')


def oscillator(num_neurons= 200, tau=9, net=None):
    if net is None:
        net = nengo.Network()
    with net:
        # net.input = nengo.Node(size_in= 1)
        net.ensemble = nengo.Ensemble(
            n_neurons= num_neurons, dimensions= 2, radius= 8)

        # osc to osc connection
        def feedback(x):
            x, y = x#, x_avg
            dx = 3 * x - x ** 3 + 2 - y + np.random.normal(0,1,None)
            dy = Constants.epsilon * (Constants.gamma * (1 + np.tanh(x / Constants.beta)) - y)
            return [tau * dx + x, tau * dy + y]#, tau*x+.01*x_avg

        nengo.Connection(net.ensemble, net.ensemble, function= feedback, synapse= tau)
        # inp to osc connection
        # nengo.Connection(net.input, net.ensemble[0], transform= tau, synapse= tau)
    return net


def test_oscillator(net,osc,time=5, vectorfield = False):
    # ex: test_oscillator(net,net.oscB)
    with net:
        piecewise_f = piecewise({0:-1,
                                 0.2:0})
        piecewise_inp = nengo.Node(1)
        nengo.Connection(piecewise_inp, osc.input)
        input_probe = nengo.Probe(piecewise_inp)
        x_pr = nengo.Probe(osc.ensemble[0], synapse=0.01)
        y_pr = nengo.Probe(osc.ensemble[1], synapse=0.01)
        x_avg_pr = nengo.Probe(osc.ensemble[2], synapse=0.01)
    with nengo.Simulator(net) as sim:
        sim.run(time)
    # <editor-fold desc="...plot">
    t = sim.trange()
    # xy activities
    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.plot(t, sim.data[input_probe], label="input", color='k')
    ax1.plot(t, sim.data[x_pr], label="x activity", color='b')
    ax1.plot(t, sim.data[y_pr], label="y activity", color='r')
    ax1.plot(t, sim.data[x_avg_pr], label="x_avg", color='#3d6305')
    ax1.set_title("Activities of the Oscillator")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("x&y activities")
    ax1.set_ylim(-3, 7)
    ax1.legend()

    # phase plane
    xmin, xmax, ymin, ymax = -2.5, 2.5, -2, 8
    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.plot(sim.data[x_pr], sim.data[y_pr], label="Neuron Activity", color='#ffa500', linewidth=.5, marker='x')
    X = np.linspace(xmin, xmax)
    ax2.plot(X, 3. * X - X ** 3 + 2 + Constants.I, label='x-nullcline', color='b')
    ax2.plot(X, Constants.gamma * (1 + np.tanh(X / Constants.beta)), label='y-nullcline', color='r')
    if vectorfield==True:
        vecterfield = ["3 * X - X**3 + 2 - Y",
            "Constants.epsilon * (Constants.gamma * (1 + np.tanh(X / Constants.beta)) - Y)"]
        VectorField(ax2, vecterfield, xran=[xmin, xmax], yran=[ymin, ymax])
    ax2.set_title("Phase Plane")
    ax2.set_xlabel("x activity")
    ax2.set_ylabel("y activity")
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlim(xmin, xmax)
    ax2.legend()
    #
    # plt.grid()
    plt.show()
    # prettify()
    # plt.legend(['$x_0$', '$x_1$', '$\omega$']);
    # </editor-fold>


def inhibitor(num_neurons=150, tau=2.5, net=None):
    if net is None:
        net = nengo.Network()
    with net:
        net.input = nengo.Node(size_in=1)#, output=lambda t, x: 1 if x >= .9 else 0
        net.ensemble = nengo.Ensemble(n_neurons=num_neurons,
                                      dimensions=1,
                                      radius=1)
        nengo.Connection(net.input, net.ensemble, transform= tau* Constants.phi, synapse=tau)
        nengo.Connection(net.ensemble, net.ensemble, transform= [-tau*Constants.phi+1], synapse=tau)
    return net


def testinhib(net, inhib, T=10, dt=0.001):
    with net:
        piecewise_f = piecewise({0:1,
                                 5:0,
                                 10:1,
                                 15:0,
                                 20:3,
                                 25:0})
        piecewise_inp = nengo.Node(piecewise_f)
        nengo.Connection(piecewise_inp, inhib.input)
        piecewise_inp_probe = nengo.Probe(piecewise_inp,synapse=0.01)
        input_probe = nengo.Probe(inhib.input, synapse=0.01)
        z_pr = nengo.Probe(inhib.ensemble, synapse=0.01)
    with nengo.Simulator(net,dt=dt) as sim:
        sim.run(T)
    # <editor-fold desc="...plot">
    t = sim.trange()
    # z activities
    plt.figure()
    plt.plot(t, sim.data[piecewise_inp_probe], label="piecewise_inp_probe", color='r')
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


def s_f(x, theta):
    s = 1 / (1 + np.exp(-Constants.kappa * (x - theta)))
    return s

def inhib2local_connect(num_neurons, Inh, Local, tau):
    #connection from inhib to local oscillators

    def weight(x):
        # s = 1 / (1 + np.exp(-Constants.kappa * (x - Constants.theta_1)))
        # new=s_f(z, Constants.theta_1)
        return -Constants.W1*s_f(x,Constants.theta_1)
    # def justfunction(x):
    #     return x*[-2.5 for i in range(num_neurons)]
    # nengo.Connection(Inh, Local.neurons, function=justfunction, synapse=tau)
    nengo.Connection(Inh, Local, function= weight, synapse = tau)


def local2inhib_connect(Local, Inh, tau):
    def justfun(x):
        if x >= Constants.theta_z:
            x = 1
        else:
            x = 0
        return x
    nengo.Connection(Local,Inh, function=justfun, synapse=tau)
    # nengo.Connection(Local, Inh, synapse=tau)

def local2local_connect(Local1, Local2, tau):
    nengo.Connection(Local1, Local2, transform= Constants.W0, synapse=tau)


# net = nengo.Network(label="withseed",seed=124)
net = nengo.Network(label="withoutseed")
with net:
    net.config[nengo.Ensemble].neuron_type = nengo.Direct()
    num = 1 #num doesn't matter when we are using nengo.Direct()
    tau = 2.0
#components
    net.oscA = oscillator(num, tau)
    net.oscB = oscillator(num, tau)
    net.oscC = oscillator(num, tau)
    net.oscD = oscillator(num, tau)
    # test_oscillator(net, net.oscA, 40)
    # test_oscillator(net, net.oscB, 20)

    net.inhib = inhibitor(num, tau)
    # testinhib(net, net.inhib, 40)

#connections
    #inhib
    inhib2local_connect(num, net.inhib.ensemble, net.oscA.ensemble[0], tau) #
    inhib2local_connect(num, net.inhib.ensemble, net.oscB.ensemble[0], tau)
    inhib2local_connect(num, net.inhib.ensemble, net.oscC.ensemble[0], tau)
    inhib2local_connect(num, net.inhib.ensemble, net.oscD.ensemble[0], tau)

    local2inhib_connect(net.oscA.ensemble[0], net.inhib.input,tau)
    local2inhib_connect(net.oscB.ensemble[0], net.inhib.input,tau)#
    local2inhib_connect(net.oscC.ensemble[0], net.inhib.input, tau)
    local2inhib_connect(net.oscD.ensemble[0], net.inhib.input, tau)

    #local
    local2local_connect(net.oscA.ensemble[0], net.oscB.ensemble[0], tau)
    # local2local_connect(net.oscA.ensemble[0], net.oscC.ensemble[0], tau)
    # local2local_connect(net.oscA.ensemble[0], net.oscD.ensemble[0], tau)

    local2local_connect(net.oscB.ensemble[0], net.oscA.ensemble[0], tau)
    local2local_connect(net.oscB.ensemble[0], net.oscC.ensemble[0], tau)
    # local2local_connect(net.oscB.ensemble[0], net.oscD.ensemble[0], tau)

    # local2local_connect(net.oscC.ensemble[0], net.oscA.ensemble[0], tau)
    local2local_connect(net.oscC.ensemble[0], net.oscB.ensemble[0], tau)
    local2local_connect(net.oscC.ensemble[0], net.oscD.ensemble[0], tau)

    # local2local_connect(net.oscD.ensemble[0], net.oscA.ensemble[0], tau)
    # local2local_connect(net.oscD.ensemble[0], net.oscB.ensemble[0], tau)
    local2local_connect(net.oscD.ensemble[0], net.oscC.ensemble[0], tau)

#feed
    piecewise_f = piecewise({0: 1,
                             .2: 1})
    net.A = nengo.Node(1)
    net.B = nengo.Node(0)
    net.C = nengo.Node(0)
    net.D = nengo.Node(1)
    nengo.Connection(net.A, net.oscA.ensemble[0], synapse=tau)
    nengo.Connection(net.B, net.oscB.ensemble[0], synapse=tau)
    nengo.Connection(net.C, net.oscC.ensemble[0], synapse=tau)
    nengo.Connection(net.D, net.oscD.ensemble[0], synapse=tau)
#probes
    inhib_pr = nengo.Probe(net.inhib.ensemble, synapse=0.01)
    oscillatorA_pr = nengo.Probe(net.oscA.ensemble[0], synapse=0.01)
    oscillatorB_pr = nengo.Probe(net.oscB.ensemble[0], synapse=0.01)
    oscillatorC_pr = nengo.Probe(net.oscC.ensemble[0], synapse=0.01)
    oscillatorD_pr = nengo.Probe(net.oscD.ensemble[0], synapse=0.01)


with nengo.Simulator(net) as sim:
    sim.run(100)
    t = sim.trange()
    plt.figure()
    plt.subplot(5,1,1)
    plt.plot(t, sim.data[inhib_pr], label="inhibitor", color='k')

    plt.subplot(5,1,2)
    plt.plot(t, sim.data[oscillatorA_pr], label="oscillatorA", color='b')

    plt.subplot(5,1,3)
    plt.plot(t, sim.data[oscillatorB_pr], label="oscillatorB", color='b')

    plt.subplot(5,1,4)
    plt.plot(t, sim.data[oscillatorC_pr], label="oscillatorC", color='b')

    plt.subplot(5,1,5)
    plt.plot(t, sim.data[oscillatorD_pr], label="oscillatorD", color='b')

    # plt.title("osc and inhib interaction")
    # plt.xlabel("Time (s)")
    # plt.ylabel("x&z activities")
    # plt.legend()
    plt.show()
