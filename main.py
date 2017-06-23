from vfield import VectorField
import matplotlib.pyplot as plt
from Constants import *
import numpy as np
import nengo
from nengo.utils.functions import piecewise
# plt.close('all')

class Movingavg(object):
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((timesteps, dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        if np.mean(self.history)>theta:
            temp = 1
        else:
            temp = 0
        return temp

class Oscillator(nengo.Network):
    def __init__(self, num_neurons= 1, tau= default_tau, syn = default_syn,
                 x_t=0 ,x_f = 0,
                 label=None):
        super().__init__()
        self.n_neuron = num_neurons
        self.radius = 8
        self.x_t = x_t
        self.x_f = x_f
        self.syn = syn
        self.count = 0
        with self:
            # self.config[nengo.Ensemble].neuron_type = nengo.Direct()
            # self.input = nengo.Node(size_in= 1)
            self.ensemble = nengo.Ensemble(
                n_neurons= self.n_neuron, dimensions= 2, radius= self.radius , label=label)

            # osc to osc connection
            def feedback(x):
                x, y = x
                dx = 3 * x - x ** 3 + 2 - y #+ np.random.normal(0,1,None)
                dy = epsilon * (gamma * (1 + np.tanh(x / beta)) - y)
                return [tau * dx + x, tau * dy + y]#, tau*x+.01*x_avg

            nengo.Connection(self.ensemble, self.ensemble, function= feedback, synapse= self.syn)

            #moving average
            deltat = t_th #window size for moving average
            movingavg = Movingavg(1, timesteps=int(deltat / dt))
            self.h = nengo.Node(movingavg.step, size_in=1, size_out=1)
            nengo.Connection(self.ensemble[0], self.h)

            #NJ ensemble
            self.NJensemble = nengo.Ensemble(
                n_neurons=self.n_neuron, dimensions=8, radius=self.radius)

            def nj_osc(z):
                Ji1, Ji2, Ji3, Ji4, x1, x2, x3, x4 = z
                sum = c + Ji1 + Ji2 + Ji3 + Ji4
                return [W_T * Ji1 / sum * s_f(x1, theta_x) + W_T * Ji2 / sum * s_f(x2, theta_x) +
                        W_T * Ji3 / sum * s_f(x3, theta_x) + W_T * Ji4 / sum * s_f(x4, theta_x)]

            nengo.Connection(self.NJensemble, self.ensemble[0], function=nj_osc, synapse=self.syn)

    def addconnection(self, Jij, x):
        if self.count<4:
            nengo.Connection(Jij, self.NJensemble[self.count], synapse= self.syn)
            nengo.Connection(x, self.NJensemble[self.count+4], synapse= self.syn)
            self.count= self.count+1
        else:
            print(self.count)
            print("error")

def inhibitor(num_neurons=150, tau= default_tau, model=None):
    if model is None:
        model = nengo.Network()
    with model:
        model.input = nengo.Node(size_in=1, output=lambda t, x: 1 if x >= .9 else 0)
        #, output=lambda t, x: 1 if x >= .9 else 0
        model.ensemble = nengo.Ensemble(n_neurons=num_neurons,
                                        dimensions=1,
                                        radius=1)
        nengo.Connection(model.input, model.ensemble, transform=tau * phi, synapse=tau)
        nengo.Connection(model.ensemble, model.ensemble, transform= [-tau * phi + 1], synapse=tau)
    return model


def s_f(x, theta):
    s = 1 / (1 + np.exp(-kappa * (x - theta)))
    return s


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


def T_f(xi, xj):
    W_permanent = np.exp(-((xj.x_t - xi.x_t)**2 / sigma_t**2 + (xj.x_f - xi.x_f)**2 / sigma_f**2))
    return W_permanent

# <editor-fold desc="...simplified local connection function">
# def local2local_connect(Local1, Local2, tau):
#     nengo.Connection(Local1, Local2, transform= W0, synapse=tau)
# </editor-fold>

def Jij_connector(xi, xj, tau=default_tau, syn=default_syn, model=None):
    if model is None:
        model = nengo.Network()
    with model:
        model.label = "R" + str(xi.x_f) + "C" + str(xi.x_t) + "-R" + str(xj.x_f) + "C" + str(xj.x_t)
        model.n_neuron = 1
        model.radius = 8
        model.jensemble = nengo.Ensemble(
            n_neurons=model.n_neuron, dimensions=3, radius=model.radius, label= model.label)
        nengo.Connection(xi.h, model.jensemble[0], synapse= syn)
        nengo.Connection(xj.h, model.jensemble[1], synapse= syn)
        def jfeedback(x):
            h_xi, h_xj, Jij = x
            delta_J = eta * T_f(xi, xj) * h_xi * h_xj
            return [delta_J * tau + Jij]
        nengo.Connection(model.jensemble, model.jensemble[2], function=jfeedback, synapse=syn)
    return model


# net = nengo.Network(label="withseed",seed=124)
model = nengo.Network(label="withoutseed")
dt = 0.001
deltat = 0.2
movingavg = Movingavg(1, timesteps=int(deltat/ dt))
with model:
    model.config[nengo.Ensemble].neuron_type = nengo.Direct()# force direct
    num = 1 #num doesn't matter when we are using nengo.Direct()
    tau = 3#3
    syn = tau
    inhib = inhibitor(num, tau)


    grid_r, grid_c = inp.shape
    ea_oscillator = [[0 for x in range(grid_c)] for x in range(grid_r)]


    # create input nodes, oscillators, and
    # oscillators and inhib connections
    for i in range(grid_r):
        for j in range(grid_c):
            inputnode = nengo.Node(inp[i][j], label="input%d%d" % (i, j))
            ea_oscillator[i][j] = Oscillator(num, tau, syn, i, j)
            nengo.Connection(inputnode,
                             ea_oscillator[i][j].ensemble[0],
                             synapse = tau)
            inhib2local_connect(inhib.ensemble,
                                ea_oscillator[i][j].ensemble[0],
                                tau)
            local2inhib_connect(ea_oscillator[i][j].ensemble[0],
                                inhib.input,
                                tau)



    # <editor-fold desc="...simplified local connections">
    # for i in range(grid_r):
    #     for j in range(grid_c):
    #         if i - 1 >= 0:
    #             local2local_connect(model.ea_oscillator[i][j].ensemble[0],
    #                                 model.ea_oscillator[i - 1][j].ensemble[0],
    #                                 model.tau)
    #         if i + 1 < grid_r:
    #             local2local_connect(model.ea_oscillator[i][j].ensemble[0],
    #                                 model.ea_oscillator[i + 1][j].ensemble[0],
    #                                 model.tau)
    #         if j - 1 >= 0:
    #             local2local_connect(model.ea_oscillator[i][j].ensemble[0],
    #                                 model.ea_oscillator[i][j - 1].ensemble[0],
    #                                 model.tau)
    #         if j + 1 < grid_c:
    #             local2local_connect(model.ea_oscillator[i][j].ensemble[0],
    #                                 model.ea_oscillator[i][j + 1].ensemble[0],
    #                                 model.tau)
    # </editor-fold>

    # local connection
    ea_Jconnector = []

    for i in range(0, grid_r):
        for j in range(0, grid_c):
            if i + 1 < grid_r:
                J = Jij_connector(ea_oscillator[i][j],
                                        ea_oscillator[i + 1][j],
                                        tau, syn)
                ea_oscillator[i][j].addconnection(J.jensemble[2],
                                                        ea_oscillator[i + 1][j].ensemble[0])
                ea_oscillator[i + 1][j].addconnection(J.jensemble[2],
                                                            ea_oscillator[i][j].ensemble[0])
                ea_Jconnector.append(J)
                # print("connector between (",i,j, ") and (",i+1,j,") constructed")
                # nengo.Connection(J.jensemble[2], ea_oscillator[i][j].NJensemble, synapse=syn)
                # nengo.Connection(J.jensemble[2], ea_oscillator[i][j].NJensemble, synapse=syn)
            if j + 1 < grid_c:
                J = Jij_connector(ea_oscillator[i][j],
                                  ea_oscillator[i][j + 1],
                                  tau, syn)
                ea_oscillator[i][j].addconnection(J.jensemble[2],
                                                        ea_oscillator[i][j + 1].ensemble[0])
                ea_oscillator[i][j + 1].addconnection(J.jensemble[2],
                                                            ea_oscillator[i][j].ensemble[0])
                ea_Jconnector.append(J)
                # print("connector between (", i, j, ") and (", i, j+1, ") constructed")


    # print("total number of Jconnector= " + str(len(ea_Jconnector)))
    # for i in ea_Jconnector:
    #     print(str(J.label))

#probes

    inhibitor_probe = nengo.Probe(inhib.ensemble, synapse= 0.01)
    oscillator_probes = [[0 for x in range(grid_c)] for x in range(grid_r)]
    for i in range(grid_r):
        for j in range(grid_c):
            oscillator_probes[i][j] = nengo.Probe(ea_oscillator[i][j].ensemble[0], synapse=0.01)


with nengo.Simulator(model) as sim:
    sim.run(runtime)
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
