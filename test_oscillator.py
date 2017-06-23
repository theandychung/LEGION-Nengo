import nengo
import numpy as np
import matplotlib.pyplot as plt
from Constants import *

# <editor-fold desc="...constants">
# I =.5
# epsilon=.2
# gamma=6.0
# beta=0.1
# dt = 0.001
# </editor-fold>

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
    def __init__(self, num_neurons= 1, tau=4, syn = 0.2,
                 x_t=0 ,x_f = 0,
                 label=None):
        super().__init__()
        self.n_neuron = num_neurons
        self.radius = 8
        self.x_t = x_t
        self.x_f = x_f
        self.syn = syn
        self.jcount = 0
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
            
def s_f(x, theta):
    s = 1 / (1 + np.exp(-kappa * (x - theta)))
    return s

model = nengo.Network(label='Relaxation Oscillator')
with model:
    model.config[nengo.Ensemble].neuron_type = nengo.Direct()# force direct
    # net.syn = .05
    # net.tau = .5
    model.syn = 1
    model.tau = 1
    model.inp = nengo.Node(I)
    model.osc = Oscillator(500, model.tau, model.syn)
    nengo.Connection(model.inp, model.osc.ensemble[0], synapse= model.syn)

    x_pr = nengo.Probe(model.osc.ensemble[0], synapse=0.01)
    y_pr = nengo.Probe(model.osc.ensemble[1], synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(40)
# <editor-fold desc="...plot">
t = sim.trange()
# xy activities
fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(1, 2, 1)
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