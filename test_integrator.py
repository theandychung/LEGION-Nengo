import matplotlib.pyplot as plt
import nengo
from nengo.utils.functions import piecewise

model = nengo.Network(label='Integrator')
with model:
    # Our ensemble consists of 100 leaky integrate-and-fire neurons,
    # representing a one-dimensional signal
    A = nengo.Ensemble(100, dimensions=1, neuron_type=nengo.Direct())

    input = nengo.Node(
        piecewise({
            0: -2,
            .2:2,
            1: -2,
            1.2:2,
            2: -2,
            2.2:2,
            3: -2,
            3.2:2,
            4: -2,
            4.2:2,
            5: -2
        }))
    # Connect the population to itself
    tau = 0.1
    nengo.Connection(
        A, A, transform=.8,
        synapse=tau)  # Using a long time constant for stability

    # Connect the input
    nengo.Connection(
        input, A, transform=[[tau]], synapse=tau
    )  # The same time constant as recurrent to make it more 'ideal'

    # Add probes
    input_probe = nengo.Probe(input)
    A_probe = nengo.Probe(A, synapse=0.01)

    with nengo.Simulator(model) as sim:
        # Run it for 6 seconds
        sim.run(6)

    plt.figure()
    plt.plot(sim.trange(), sim.data[input_probe], label="Input")
    plt.plot(sim.trange(), sim.data[A_probe], 'k', label="Integrator output")
    plt.legend()
    plt.show()