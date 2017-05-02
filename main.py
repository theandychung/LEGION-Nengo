#<editor-fold desc="...import">
from vfield import VectorField
import matplotlib.pyplot as plt
import numpy as np
import Constants
#</editor-fold>
# plt.close('all')

import nengo

tau=.6
syn = 0.05

model = nengo.Network(label='TW Oscillator')
with model:
    inp = nengo.Node(Constants.I)
    oscillator = nengo.Ensemble(n_neurons= 150,
                                dimensions=2,
                                radius=4,
                                neuron_type= nengo.LIFRate(tau_rc=0.02, tau_ref=0.002))
                                #max_rates=[100])
     # osc to osc connection
    def feedback(x):
        x,y = x
        dx =  3 * x - x**3 + 2 - y
        dy = Constants.epsilon * (Constants.gamma * (1 + np.tanh(x / Constants.beta)) - y)
        return [tau*dx+x,tau*dy+y]
    nengo.Connection(oscillator, oscillator, function=feedback, synapse=syn)
    # inp to osc connection
    nengo.Connection(inp,oscillator[0], transform = tau)
############################################################################################
    x_pr = nengo.Probe(oscillator[0],synapse=syn)
    # xspikes_pr = nengo.Probe(oscillator.neurons,'spikes')
    y_pr = nengo.Probe(oscillator[1], synapse=syn)

with nengo.Simulator(model) as sim:
    sim.run(3)



#<editor-fold desc="...plot">
t = sim.trange()
# xy activities
fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(1,2,1)
ax1.plot(t,sim.data[x_pr], label="x activity", color='b')
ax1.plot(t,sim.data[y_pr], label="y activity", color='r')
ax1.set_title("Activities of the Oscillator")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("x&y activities")
ax1.set_ylim(-3, 7)
ax1.legend()

# phase plane
xmin, xmax, ymin, ymax = -2.5, 2.5, -2, 8
ax2 = fig1.add_subplot(1,2,2)
ax2.plot(sim.data[x_pr],sim.data[y_pr], label="Neuron Activity", color='#ffa500', linewidth=.5, marker='x')
X = np.linspace(xmin,xmax)
ax2.plot(X,3.*X-X**3+2+Constants.I,label='x-nullcline',color='b')
ax2.plot(X,Constants.gamma * (1 + np.tanh(X / Constants.beta)),label='y-nullcline',color='r')

vecterfield = ["3 * X - X**3 + 2 - Y", "Constants.epsilon * (Constants.gamma * (1 + np.tanh(X / Constants.beta)) - Y)"]
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
#prettify()
#plt.legend(['$x_0$', '$x_1$', '$\omega$']);
#</editor-fold>