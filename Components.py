import nengo
import Constants
import numpy as np


def Inhibitor():
    with nengo.Network() as inhibitor:
        inhibitor.input = nengo.Node(size_in=1)
        inhibitor.ensemble = nengo.
    return inhibitor

def Oscillator():
    tau = .6
    syn = 0.05
    with nengo.Network() as oscillator:
        oscillator.input = nengo.Node(size_in=1)
        oscillator.ensemble = nengo.Ensemble(n_neurons=150,
                                    dimensions=2,
                                    radius=4,
                                    neuron_type=nengo.LIFRate(tau_rc=0.02, tau_ref=0.002))

        # max_rates=[100])
        # osc to osc connection
        def feedback(x):
            x, y = x
            dx = 3 * x - x ** 3 + 2 - y
            dy = Constants.epsilon * (Constants.gamma * (1 + np.tanh(x / Constants.beta)) - y)
            return [tau * dx + x, tau * dy + y]

        nengo.Connection(oscillator.ensemble, oscillator.ensemble, function=feedback, synapse=syn)
        # inp to osc connection
        nengo.Connection(oscillator.input, oscillator.ensemble[0], transform=tau)
    return oscillator
