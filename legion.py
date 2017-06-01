import nengo
from nengo.utils.compat import is_iterable, range
from Constants import *


class legion(nengo.Network):
    def __init__(self, n_neurons = 1, tau = 2.0,
                 label=None, seed=None):

        #super(legion, self).__init__(label, seed, add_to_container)

        self.n_neurons = n_neurons
        self.Inp = inp
        self.tau = tau

        def oscillator(self, num_neurons=200, tau=9, net=None):
            if net is None:
                net = nengo.Network()
            with net:
                # net.input = nengo.Node(size_in= 1)
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

        def inhibitor(self, num_neurons=150, tau=2.5, net=None):
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

        def s_f(self, x, theta):
            s = 1 / (1 + np.exp(-kappa * (x - theta)))
            return s

        def inhib2local_connect(self, Inh, Local, tau):
            # connection from inhib to local oscillators
            def weight(x):
                return -W1 * self.s_f(x, theta_1)

            # def justfunction(x):
            #     return x*[-2.5 for i in range(num_neurons)]
            # nengo.Connection(Inh, Local.neurons, function=justfunction, synapse=tau)
            nengo.Connection(Inh, Local, function=weight, synapse=tau)

        def local2inhib_connect(self, Local, Inh, tau):
            def justfun(x):
                if x >= theta_z:
                    x = 1
                else:
                    x = 0
                return x

            nengo.Connection(Local, Inh, function=justfun, synapse=tau)
            # nengo.Connection(Local, Inh, synapse=tau)

        def local2local_connect(self, Local1, Local2, tau):
            nengo.Connection(Local1, Local2, transform=W0, synapse=tau)

        grid_r, grid_c = self.Inp.shape
        self.ea_oscillator = [[0 for x in range(grid_c)] for x in range(grid_r)]


        with self:
            self.config[nengo.Ensemble].neuron_type = nengo.Direct() #force direct
            self.inhib = inhibitor(n_neurons, self.tau)


            #create input nodes, oscillators, and
            #establish the connection between oscillators and inhib
            for i in range(grid_r):
                for j in range(grid_c):
                    self.inputnode = nengo.Node(self.Inp[i][j], label="input%d%d" % (i, j))
                    self.ea_oscillator[i][j] = oscillator(self.n_neurons, self.tau)
                    inhib2local_connect(self.n_neurons, self.inhib.ensemble,
                                        self.ea_oscillator[i][j].ensemble[0], self.tau)
                    local2inhib_connect(self.ea_oscillator[i][j].ensemble[0],
                                        self.inhib.input, self.tau)


            for i in range(grid_r):
                for j in range(grid_c):
                    if i - 1 >= 0:
                        local2local_connect(self.ea_oscillator[i][j].ensemble[0],
                                            self.ea_oscillator[i - 1][j].ensemble[0],
                                            self.tau)
                    if i + 1 < grid_r:
                        local2local_connect(self.ea_oscillator[i][j].ensemble[0],
                                            self.ea_oscillator[i + 1][j].ensemble[0],
                                            self.tau)
                    if j - 1 >= 0:
                        local2local_connect(self.ea_oscillator[i][j].ensemble[0],
                                            self.ea_oscillator[i][j - 1].ensemble[0],
                                            self.tau)
                    if j + 1 < grid_c:
                        local2local_connect(self.ea_oscillator[i][j].ensemble[0],
                                            self.ea_oscillator[i][j + 1].ensemble[0],
                                            self.tau)

net = nengo.Network(label="withoutseed")
with net:
    net.obj = legion(1,2.0)

    # obj_pr = nengo.Probe(net.obj.ea_oscillator[1])