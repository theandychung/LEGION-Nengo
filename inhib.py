
from Constants import *
import nengo
from func import *


class Inhibitor(nengo.Network):
    def __init__(self, tau= default_tau, syn= default_syn):
        super().__init__()

        self.n_neuron = 1
        self.tau = tau
        self.syn = syn
        self.label = "Global Inhibitor"
        #, output=lambda t, x: 1 if x >= .9 else 0
        with self:
            self.input = nengo.Node(size_in=1, output=lambda t, x: 1 if x >= .1 else 0)#
            self.ensemble = nengo.Ensemble(n_neurons=self.n_neuron,
                                            dimensions=1,
                                            radius=1,
                                           label="inhib bone")
            nengo.Connection(self.input, self.ensemble, transform=self.tau * phi, synapse=self.syn)
            nengo.Connection(self.ensemble, self.ensemble, transform= [-self.tau * phi + 1], synapse=self.syn)

    def inhiblocal_connect(self, Local):
        #connection from inhib to local oscillators
        def weight(x):
            return -W1*s_f(x,theta_1)
        # def justfunction(x):
        #     return x*[-2.5 for i in range(num_neurons)]
        # nengo.Connection(Inh, Local.neurons, function=justfunction, synapse=tau)
        nengo.Connection(self.ensemble, Local, function= weight, synapse = self.syn)
        def justfun(x):
            if x >= theta_z:
                x = 1
            else:
                x = 0
            return x
        nengo.Connection(Local,self.input, function=justfun, synapse=self.syn)