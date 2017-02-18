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
    X = nengo.Ensemble(100, dimensions=2)
    nengo.Connection(inp, X[0], transform=tau, synapse=tau)
    nengo.Connection(X[0], X[0], synapse=tau)

    Y = nengo.Ensemble(100, dimensions=2)
    nengo.Connection(Y[0], Y[0], synapse=tau)

    def XtoY(z):
        x, y = z
        dy = eta * (gamma * (1 + math.tanh(x / beta)) - y)
        return x, dy * tau
    nengo.Connection(X, Y, function=XtoY)

    def YtoX(z):
        y, x = z
        dx = 3 * x - x**3 + 2 - y
        return dx * tau, y
    nengo.Connection(X, Y, function=YtoX)