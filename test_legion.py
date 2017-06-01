import numpy as np
import nengo

I = .5
epsilon = .2
gamma = 6.0
beta = 0.1

# epsilon = .02;
# gamma = 6.0;
# beta = 0.1;

W0 = 0.1  # weight for local connection (temp)

# inp = np.array([[1,2,3],
#               [4,5,6]])
inp = np.array([[1, 0],
                [0, 0]])
rho = .02  # amplitude of gaussian noise
phi = 4.0  # the rate at which the inhibitor reacts to the stimulation.
W1 = 9.0
W2 = 0
theta = 0.05  # x is enabled? if x>theta_sp, h(x)=1;else h(x)=0;
theta_x = -.2
theta_1 = 0.5
theta_z = -.5
kappa = 50
t_th = 7  # for <x>
num_z = 2  # global inhibitor
eta = 10  # control speed of DJ
W_T = 6  # DJ weights
sigma_t = 8.0
sigma_f = 5.0


def oscillator(num_neurons=200, tau=9, net=None):
    if net is None:
        net = nengo.Network()
    with net:
        net.ensemble = nengo.Ensemble(
            n_neurons=num_neurons, dimensions=2, radius=8)

        # osc to osc connection
        def feedback(x):
            x, y = x  # , x_avg
            dx = 3 * x - x ** 3 + 2 - y + np.random.normal(0, 1, None)
            dy = epsilon * (gamma * (1 + np.tanh(x / beta)) - y)
            return [tau * dx + x, tau * dy + y]  # , tau*x+.01*x_avg

        nengo.Connection(net.ensemble, net.ensemble, function=feedback, synapse=tau)
        # inp to osc connection
        # nengo.Connection(net.input, net.ensemble[0], transform= tau, synapse= tau)
    return net


def inhibitor(num_neurons=150, tau=2.5, net=None):
    if net is None:
        net = nengo.Network()
    with net:
        net.input = nengo.Node(size_in=1)  # , output=lambda t, x: 1 if x >= .9 else 0
        net.ensemble = nengo.Ensemble(n_neurons=num_neurons,
                                      dimensions=1,
                                      radius=1)
        nengo.Connection(net.input, net.ensemble, transform=tau * phi, synapse=tau)
        nengo.Connection(net.ensemble, net.ensemble, transform=[-tau * phi + 1], synapse=tau)
    return net


def s_f(x, theta):
    s = 1 / (1 + np.exp(-kappa * (x - theta)))
    return s


def inhib2local_connect(num_neurons, Inh, Local, tau):
    # connection from inhib to local oscillators
    def weight(x):
        return -W1 * s_f(x, theta_1)

    # def justfunction(x):
    #     return x*[-2.5 for i in range(num_neurons)]
    # nengo.Connection(Inh, Local.neurons, function=justfunction, synapse=tau)
    nengo.Connection(Inh, Local, function=weight, synapse=tau)


def local2inhib_connect(Local, Inh, tau):
    def justfun(x):
        if x >= theta_z:
            x = 1
        else:
            x = 0
        return x

    nengo.Connection(Local, Inh, function=justfun, synapse=tau)
    # nengo.Connection(Local, Inh, synapse=tau)


def local2local_connect(Local1, Local2, tau):
    nengo.Connection(Local1, Local2, transform=W0, synapse=tau)


model = nengo.Network(label="withoutseed")
with model:
    model.config[nengo.Ensemble].neuron_type = nengo.Direct()
    num = 1  # num doesn't matter when we are using nengo.Direct()
    tau = 2.0
    inhib = inhibitor(num, tau)

    grid_r, grid_c = inp.shape
    ea_oscillator = [[0 for x in range(grid_c)] for x in range(grid_r)]
    ea_collector = [0 for x in range(grid_r)]
    for i in range(grid_r):
        ea_collector[i] = nengo.Ensemble(num, dimensions=grid_c, radius=1)
    # create input nodes, oscillators, and
    # establish the connection between oscillators and inhib
    for i in range(grid_r):
        for j in range(grid_c):
            inputnode = nengo.Node(inp[i][j], label="input%d%d" % (i, j))
            ea_oscillator[i][j] = oscillator(num, tau)
            nengo.Connection(inputnode,
                             ea_oscillator[i][j].ensemble[0],
                             synapse=tau)
            inhib2local_connect(num, inhib.ensemble,
                                ea_oscillator[i][j].ensemble[0],
                                tau)
            local2inhib_connect(ea_oscillator[i][j].ensemble[0],
                                inhib.input,
                                tau)
            nengo.Connection(ea_oscillator[i][j].ensemble[0],
                             ea_collector[i][j],
                             synapse=tau)

    for i in range(grid_r):
        for j in range(grid_c):
            if i - 1 >= 0:
                local2local_connect(ea_oscillator[i][j].ensemble[0],
                                    ea_oscillator[i - 1][j].ensemble[0],
                                    tau)
            if i + 1 < grid_r:
                local2local_connect(ea_oscillator[i][j].ensemble[0],
                                    ea_oscillator[i + 1][j].ensemble[0],
                                    tau)
            if j - 1 >= 0:
                local2local_connect(ea_oscillator[i][j].ensemble[0],
                                    ea_oscillator[i][j - 1].ensemble[0],
                                    tau)
            if j + 1 < grid_c:
                local2local_connect(ea_oscillator[i][j].ensemble[0],
                                    ea_oscillator[i][j + 1].ensemble[0],
                                    tau)
