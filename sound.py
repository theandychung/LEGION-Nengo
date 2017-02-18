import nengo
import math

I = 0.11
tau = 0.1
gamma = 2.1
beta = 0.1
eta = 0.35
model = nengo.Network(label='WT Oscillator')

with model:
    inp = nengo.Node(I)
    X = nengo.Ensemble(100, dimensions=1, radius=10)
    nengo.Connection(inp, X, transform=tau)

    def Xfunc(x):
        return tau*(3 * x - x**3)+x
    nengo.Connection(X, X, function=Xfunc, synapse=tau)

    Y = nengo.Ensemble(100, dimensions=1, radius=10)
    def Yfunc(y):
        return tau * -eta * y + y
    nengo.Connection(Y, Y, function=Yfunc, synapse=tau)

    def XtoY(x):
        return tau * eta * gamma * (1 + math.tanh(x / beta))
    nengo.Connection(X, Y, function=XtoY, synapse=tau)

    def YtoX(y):
        return (-y + 2) * tau
    nengo.Connection(Y, X, function=YtoX, synapse=tau)

################################################################################
#probes
    X_pr = nengo.Probe(X,synapse=tau/10)
    Y_pr = nengo.Probe(Y,synapse=tau/10)

#run
with nengo.Simulator(model) as sim:
    sim.run(8)

#plot
import matplotlib.pyplot as plt
plt.plot(sim.trange(),sim.data[X_pr], label="Oscillator x output")


#plt.plot(sim.trange(), sim.data[inp_probe], 'r', label="Input")
plt.title("Time Domain")
plt.xlabel("Time (s)")
plt.ylabel("x activity")
plt.ylim(-5, 5);
#plt.xlim(-10, 10);
plt.legend();
plt.show()
#prettify()
#plt.legend(['$x_0$', '$x_1$', '$\omega$']);