#<editor-fold desc="...import">
import matplotlib.pyplot as plt
from nengo.utils.functions import piecewise
import matplotlib.pyplot as plt
import numpy as np
#</editor-fold>

import nengo
import math

I =.1
epsilon=.2
gamma=6.0
beta=0.1
tau=.1
# I =-1
# epsilon=.2
# gamma=6.0
# beta=0.1
# tau=1
model = nengo.Network(label='TW Oscillator')
with model:
    inp = nengo.Node(I)
    oscillator = nengo.Ensemble(150, dimensions=2,radius=1,
                                neuron_type=nengo.LIFRate(tau_rc=0.02, tau_ref=0.002))
                                #max_rates=[100])
     # osc to osc connection
    def feedback(x):
        x,y = x
        dx =  3 * x - x**3 + 2 - y
        dy = epsilon * (gamma * (1 + math.tanh(x / beta)) - y)
        return [tau*dx+x,tau*dy+y]
    nengo.Connection(oscillator, oscillator, function=feedback, synapse=tau)
    # inp to osc connection
    nengo.Connection(inp,oscillator[0], transform = [[tau]])

############################################################################################
    x_pr = nengo.Probe(oscillator[0],synapse=tau/10)
    # xspikes_pr = nengo.Probe(oscillator.neurons,'spikes')
    y_pr = nengo.Probe(oscillator[1], synapse=tau / 10)


with nengo.Simulator(model) as sim:
    sim.run(25)



#<editor-fold desc="...plot">
t = sim.trange()
F1 = plt.figure(figsize=(10,5))
f11 = F1.add_subplot(1,2,1)
f11.plot(t,sim.data[x_pr], label="x activity")
f11.plot(t,sim.data[y_pr], label="y activity")
f11.set_title("Activities of the Oscillator")
f11.set_xlabel("Time (s)")
f11.set_ylabel("x&y activities")
f11.set_ylim(-3, 7)
f11.legend()

f22 = F1.add_subplot(1,2,2)
f22.plot(sim.data[x_pr],sim.data[y_pr], label="phase plane")
f22.set_xlabel("x activity")
f22.set_ylabel("y activity")
f22.set_ylim(-2, 10)
f22.set_xlim(-3, 3)
#
# F2 = plt.figure()
# plt.plot(t,sim.data[xspikes_pr])

plt.show()
#prettify()
#plt.legend(['$x_0$', '$x_1$', '$\omega$']);
#</editor-fold>