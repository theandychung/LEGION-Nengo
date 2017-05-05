

#<editor-fold desc="...import">
from vfield import VectorField
import matplotlib.pyplot as plt
import numpy as np
import Constants
import nengo
from nengo.utils.functions import piecewise
#</editor-fold>
# plt.close('all')

def Oscillator(num_neurons=500, tau=9, net=None):
    if net is None:
        net = nengo.Network()
    with net:
        net.input = nengo.Node(size_in=1)
        net.ensemble = nengo.Ensemble(n_neurons=num_neurons,
                                    dimensions=2,
                                    radius=8)

        # osc to osc connection
        def feedback(x):
            x, y = x
            dx = 3 * x - x ** 3 + 2 - y
            dy = Constants.epsilon * (Constants.gamma * (1 + np.tanh(x / Constants.beta)) - y)
            return [tau * dx + x, tau * dy + y]

        nengo.Connection(net.ensemble, net.ensemble, function=feedback, synapse=tau)
        # inp to osc connection
        nengo.Connection(net.input, net.ensemble[0], transform=tau)
    return net

def test_oscillator(net,osc,time, dt=0.001):
    # test_oscillator(net,net.oscB)
    with net:
        piecewise_f = piecewise({0:-1,
                                 0.2:0})
        piecewise_inp = nengo.Node(1)
        nengo.Connection(piecewise_inp, osc.input)
        input_probe = nengo.Probe(piecewise_inp)
        x_pr = nengo.Probe(osc.ensemble[0], synapse=0.01)
        y_pr = nengo.Probe(osc.ensemble[1], synapse=0.01)
    with nengo.Simulator(net,dt=dt,seed=122) as sim:
        sim.run(time)
    # <editor-fold desc="...plot">
    t = sim.trange()
    # xy activities
    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.plot(t, sim.data[input_probe], label="input", color='k')
    ax1.plot(t, sim.data[x_pr], label="x activity", color='b')
    ax1.plot(t, sim.data[y_pr], label="y activity", color='r')
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
                                 10:2,
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

#
# def S_f(x, theta):
#     s = 1 / (1 + np.exp(-Constants.kappa * (x - theta)))
#     return s
#
# def LocalConnect(A,B):
#     def T_f( xi_f, xi_t, xj_f, xj_t,sigma_t,sigma_f):
#         W_permanent = np.exp(-(xj_t - xi_t)**2 / Constants.sigma_t**2 + (xj_f - xi_f)**2 / Constants.sigma_f**2)
#         return W_permanent

def Inhib2LocalConnect(num_neurons,Inh,Local,tau):
    #connection from inhib to local oscillators
    # def W1(z):
    #     new=S_f(z,Constants.theta_1)
    #     return new*num_neurons
    def justfunction(x):
        return x*[-2.5 for i in range(num_neurons)]
    nengo.Connection(Inh, Local.neurons, function=justfunction, synapse=tau)

def Local2InhibConnect(Local,Inh,tau):
    def justfun(x):
        if x >= Constants.theta_z:
            x = 1
        else:
            x = 0
        return x
    nengo.Connection(Local,Inh, function=justfun, synapse=tau)

# net = nengo.Network(label="withseed",seed=124)
net = nengo.Network(label="withoutseed")
with net:
    net.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    num = 400
    tau = 9
    net.oscA = Oscillator(num,tau)
    # net.oscB = Oscillator(num, tau)
    test_oscillator(net, net.oscA, 30)
    # test_oscillator(net, net.oscB, 30)


    net.inhib = Inhibitor(num,tau)
    # testinhib(net, net.inhib, 40)

    Inhib2LocalConnect(num, net.inhib.ensemble, net.oscA.ensemble, tau)
    Inhib2LocalConnect(num, net.inhib.ensemble, net.oscB.ensemble, tau)
    Local2InhibConnect(net.oscA.ensemble[0], net.inhib.input,tau)
    Local2InhibConnect(net.oscB.ensemble[0], net.inhib.input,tau)

    oscillatorA_pr = nengo.Probe(net.oscA.ensemble[0], synapse=0.01)
    oscillatorB_pr = nengo.Probe(net.oscB.ensemble[0], synapse=0.01)
    inhib_pr = nengo.Probe(net.inhib.ensemble, synapse=0.01)


with nengo.Simulator(net,seed=123) as sim:
    sim.run(80)
    # <editor-fold desc="...plot osc inhib">
    t = sim.trange()
    plt.figure()
    plt.plot(t, sim.data[inhib_pr], label="inhibitor", color='k')
    plt.plot(t, sim.data[oscillatorA_pr], label="oscillatorA", color='b')
    plt.plot(t, sim.data[oscillatorB_pr], label="oscillatorB", color='b')
    plt.plot(t, sim.data[inhib_pr], label="inhib", color='r')
    plt.title("osc and inhib interaction")
    plt.xlabel("Time (s)")
    plt.ylabel("x&z activities")
    plt.legend()
    plt.show()
    # </editor-fold>
