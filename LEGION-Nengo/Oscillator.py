from const import *
from func import convertinp
from movingavg import Movingavg
import nengo
from func import s_f

class Oscillator(nengo.Network):
    """A local  oscillator contains only one neuron.
    When stimulated with large enough input, it will oscillate.
    All oscillators are connected to the global inhibitor, and at the same time connecting to each others.
    """
    def __init__(self, tau=default_tau, syn=default_syn,
                 x_f=0, x_t=0,**kwargs):
        super(Oscillator,self).__init__(**kwargs)
        self.n_neuron = 1
        self.radius = 8
        self.x_t = x_t
        self.x_f = x_f
        self.syn = syn
        self.count = 0
        self.label = "Local Oscillator " + str(x_f) + str(x_t)
        with self:
            self.ensemble = nengo.Ensemble(n_neurons=self.n_neuron, dimensions=2,
                                           radius=self.radius, label="osc bone")
            def feedback(x):
                # osc to osc connection
                x, y = x
                dx = 3 * x - x ** 3 + 2 - y + rho *	np.random.randn()
                dy = epsilon * (gamma * (1 + np.tanh(x / beta)) - y)
                return [tau * dx + x, tau * dy + y]
            nengo.Connection(self.ensemble, self.ensemble, function=feedback, synapse=self.syn)
    def set_input(self,input):
        """this function set a threshold for the input of oscillator"""
        with self:
            self.input = nengo.Node(convertinp(input), label="input%d%d" % (self.x_f, self.x_t))
            nengo.Connection(self.input,self.ensemble[0], synapse= self.syn)

    def local_connect(self, OtherOscillator, tau):
        """local connection from this oscillator to other oscillator"""
        def weight(x):
            """weight between oscillators"""
            return W0*s_f(x,theta_x)
        nengo.Connection(self.ensemble[0], OtherOscillator, function=weight, synapse=tau)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    model = nengo.Network(label='Relaxation Oscillator')
    with model:
        model.config[nengo.Ensemble].neuron_type = nengo.Direct()  # force direct

        tau = 4
        syn = .2

        # tau = 4
        # syn = 4

        osc = Oscillator(tau, syn)
        osc.set_input(I)

        x_pr = nengo.Probe(osc.ensemble[0], synapse=0.01)
        y_pr = nengo.Probe(osc.ensemble[1], synapse=0.01)
    with nengo.Simulator(model) as sim:
        sim.run(runtime)
    # <editor-fold desc="...plot">
    t = sim.trange()
    # xy activities

    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.plot(t, sim.data[x_pr], label="x Activity", color='b')
    ax1.plot(t, sim.data[y_pr], label="y Activity", color='r')
    ax1.set_title("Activities of the Oscillator")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("x,y Activities")
    ax1.set_ylim(-3, 7)
    ax1.legend()

    # phase plane
    xmin, xmax, ymin, ymax = -2.5, 2.5, -2, 8
    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.plot(sim.data[x_pr], sim.data[y_pr], label="Periodic Solution", color='#ffa500', linewidth=.5, marker='x')
    X = np.linspace(xmin, xmax)
    ax2.plot(X, 3. * X - X ** 3 + 2 + I, label='x-nullcline', color='b')
    ax2.plot(X, gamma * (1 + np.tanh(X / beta)), label='y-nullcline', color='r')
    ax2.set_title("Phase Plane")
    ax2.set_xlabel("x activity")
    ax2.set_ylabel("y activity")
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlim(xmin, xmax)
    ax2.legend()
    plt.show()
    # </editor-fold>