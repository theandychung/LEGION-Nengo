from const import *


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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import nengo
    from nengo.processes import WhiteSignal

    model = nengo.Network(label="Delayed connection")
    with model:
        # We'll use white noise as input
        inp = nengo.Node(WhiteSignal(2, high=5), size_out=1)
        A = nengo.Ensemble(40, dimensions=1)
        nengo.Connection(inp, A)


    # We'll make a simple object to implement the delayed connection
    class Delay(object):
        def __init__(self, dimensions, timesteps=50):
            self.history = np.zeros((timesteps, dimensions))

        def step(self, t, x):
            self.history = np.roll(self.history, -1)
            self.history[-1] = x
            print(self.history)
            return np.mean(self.history)


    dt = 0.001
    delay = Delay(1, timesteps=int(0.01 / 0.001))

    with model:
        delaynode = nengo.Node(delay.step, size_in=1, size_out=1)
        nengo.Connection(A, delaynode)

        # Send the delayed output through an ensemble
        B = nengo.Ensemble(40, dimensions=1)
        nengo.Connection(delaynode, B)

        # Probe the input at the delayed output
        A_probe = nengo.Probe(A, synapse=0.01)
        B_probe = nengo.Probe(B, synapse=0.01)
    # Run for 2 seconds
    with nengo.Simulator(model) as sim:
        sim.run(.1)
    # Plot the results
    plt.figure()
    plt.plot(sim.trange(), sim.data[A_probe], lw=2, label="Input")
    plt.plot(sim.trange(), sim.data[B_probe], lw=2, label="output")
    # plt.axvline(0.2, c='k')
    # plt.tight_layout()
    plt.legend()
    plt.show()