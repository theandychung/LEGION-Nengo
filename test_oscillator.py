import nengo
from osc import Oscillator
import matplotlib.pyplot as plt
from Constants import *
from vfield import VectorField
from movingavg import Movingavg
from func import s_f
# <editor-fold desc="...constants">
# I =.5
# epsilon=.2
# gamma=6.0
# beta=0.1
# dt = 0.001
# </editor-fold>


model = nengo.Network(label='Relaxation Oscillator')
with model:
    model.config[nengo.Ensemble].neuron_type = nengo.Direct()# force direct

    tau = 4
    syn = .2

    # tau = 4
    # syn = 4

    osc = Oscillator(tau, syn)
    osc.set_input(I)

    x_pr = nengo.Probe(osc.ensemble[0], synapse=0.01)
    y_pr = nengo.Probe(osc.ensemble[1], synapse=0.01)
    h_pr = nengo.Probe(osc.h, synapse = 0.01)
with nengo.Simulator(model) as sim:
    sim.run(runtime)
# <editor-fold desc="...plot">
t = sim.trange()
# xy activities
fig1 = plt.figure(figsize=(10, 5))
ax1 = fig1.add_subplot(1, 2, 1)
ax1.plot(t, sim.data[x_pr], label="x activity", color='b')
ax1.plot(t, sim.data[y_pr], label="y activity", color='r')
ax1.plot(t, sim.data[h_pr], label="h activity", color='r')
ax1.set_title("Activities of the Oscillator")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("x,y,h activities")
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