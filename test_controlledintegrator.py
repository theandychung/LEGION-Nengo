import numpy as np
import matplotlib.pyplot as plt
from nengo.utils.functions import piecewise
import nengo
model = nengo.Network(label='Controlled Integrator')
with model:
    # Make a population with 225 LIF neurons representing a 2 dimensional signal,
    # with a larger radius to accommodate large inputs
    A = nengo.Ensemble(225, dimensions=2, radius=1.5,neuron_type=nengo.Direct())

    # Create a piecewise step function for input
    input_func = piecewise({
        0: 0,
        0.2: 5,
        0.3: 0,
        0.44: -10,
        0.54: 0,
        0.8: 5,
        0.9: 0
    })

    # Define an input signal within our model
    inp = nengo.Node(input_func)

    # Connect the Input signal to ensemble A.
    # The `transform` argument means "connect real-valued signal "Input" to the
    # first of the two input channels of A."
    tau = 0.1
    nengo.Connection(inp, A, transform=[[tau], [0]], synapse=tau)

    # Another piecewise step that changes half way through the run
    control_func = piecewise({0: 1, 0.2: 0.5, 0.6:1})

    control = nengo.Node(output=control_func)

    # Connect the "Control" signal to the second of A's two input channels.
    nengo.Connection(control, A[1], synapse=0.005)

    # Create a recurrent connection that first takes the product
    # of both dimensions in A (i.e., the value times the control)
    # and then adds this back into the first dimension of A using
    # a transform
    nengo.Connection(
        A, A[0], function=lambda x: tau * x[0] * x[1] + x[0],
        synapse=tau)

    # Record both dimensions of A
    A_probe = nengo.Probe(A, 'decoded_output', synapse=0.01)

with nengo.Simulator(model) as sim:  # Create a simulator
    sim.run(1.4)  # Run for 1.4 seconds

t = sim.trange()
dt = t[1] - t[0]
input_sig = list(map(input_func, t))
control_sig = list(map(control_func, t))
ref = dt * np.cumsum(input_sig)

plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1)
plt.plot(t, input_sig, label='Input')
plt.xlim(right=t[-1])
plt.ylim(-11, 11)
plt.ylabel('Input')
plt.legend(loc="lower left", frameon=False)

plt.subplot(2, 1, 2)
plt.plot(t, ref, 'k--', label='Exact')
plt.plot(t, sim.data[A_probe][:, 0], label='A (value)')
plt.plot(t, sim.data[A_probe][:, 1], label='A (control)')
plt.xlim(right=t[-1])
plt.ylim(-1.1, 1.1)
plt.xlabel('Time (s)')
plt.ylabel('x(t)')
plt.legend(loc="lower left", frameon=False)

plt.show()