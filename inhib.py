
from const import *
import nengo
from func import *


class Inhibitor(nengo.Network):
    def __init__(self, tau= default_tau, syn= default_syn,**kwargs):
        super(Inhibitor,self).__init__(**kwargs)

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


if __name__ == '__main__':
    import nengo
    import matplotlib.pyplot as plt
    from nengo.utils.functions import piecewise


    model = nengo.Network(label="inhibitor")
    with model:
        model.config[nengo.Ensemble].neuron_type = nengo.Direct()
        num = 40
        tau = .9
        T = 30
        inhib = Inhibitor(num, tau)
        piecewise_f = piecewise({0: 1,
                                 5: 0,
                                 10: .5,
                                 15: 0,
                                 20: 3,
                                 25: 0})
        piecewise_inp = nengo.Node(piecewise_f)
        nengo.Connection(piecewise_inp, inhib.input)
        piecewise_inp_probe = nengo.Probe(piecewise_inp, synapse=0.01)
        input_probe = nengo.Probe(inhib.input, synapse=0.01)
        z_pr = nengo.Probe(inhib.ensemble, synapse=0.01)
    with nengo.Simulator(model) as sim:
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
