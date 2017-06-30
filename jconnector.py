from func import *
import nengo


class Jij_connector(object):
    def __init__(self,xi, xj, tau=default_tau, syn=default_syn):
        super().__init__()
        self.label = "R" + str(xi.x_f) + "C" + str(xi.x_t) + "-R" + str(xj.x_f) + "C" + str(xj.x_t)
        self.n_neuron = 1
        self.radius = 8
        # with self:
        self.jensemble = nengo.Ensemble(
            n_neurons=self.n_neuron, dimensions=3, radius=self.radius, label= self.label)
        nengo.Connection(xi.h, self.jensemble[0], synapse= syn)
        nengo.Connection(xj.h, self.jensemble[1], synapse= syn)
        def jfeedback(x):
            h_xi, h_xj, Jij = x
            delta_J = eta * T_f(xi, xj) * h_xi * h_xj
            return [delta_J * tau + Jij]
        nengo.Connection(self.jensemble, self.jensemble[2], function=jfeedback, synapse=syn)