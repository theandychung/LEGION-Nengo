
import matplotlib.pyplot as plt
from Constants import *
import numpy as np
import nengo
from nengo.utils.functions import piecewise

from func import *
from osc import Oscillator
from jconnector import Jij_connector
from inhib import Inhibitor
# plt.close('all')

# <editor-fold desc="...simplified local connection function">
# def local2local_connect(Local1, Local2, tau):
#     nengo.Connection(Local1, Local2, transform= W0, synapse=tau)
# </editor-fold>


model = nengo.Network(label="withoutseed")
# dt = 0.001
# deltat = 0.2
# movingavg = Movingavg(1, timesteps=int(deltat/ dt))
with model:
    model.config[nengo.Ensemble].neuron_type = nengo.Direct()# force direct
    # tau = 4
    # syn = tau
    tau = 4
    syn = .2

    inh = Inhibitor(tau, syn)

    # create input nodes, oscillators, and
    # oscillators and inhib connections
    ea_oscillator = [[0 for x in range(grid_c)] for x in range(grid_r)]
    for i in range(grid_r):
        for j in range(grid_c):
            ea_oscillator[i][j] = Oscillator(tau, syn, i, j)
            ea_oscillator[i][j].set_input(inp[i][j])
            inh.inhiblocal_connect(ea_oscillator[i][j].ensemble[0])


    # <editor-fold desc="...simplified local connections">
    # for i in range(grid_r):
    #     for j in range(grid_c):
    #         if i - 1 >= 0:
    #             local2local_connect(model.ea_oscillator[i][j].ensemble[0],
    #                                 model.ea_oscillator[i - 1][j].ensemble[0],
    #                                 model.tau)
    #         if i + 1 < grid_r:
    #             local2local_connect(model.ea_oscillator[i][j].ensemble[0],
    #                                 model.ea_oscillator[i + 1][j].ensemble[0],
    #                                 model.tau)
    #         if j - 1 >= 0:
    #             local2local_connect(model.ea_oscillator[i][j].ensemble[0],
    #                                 model.ea_oscillator[i][j - 1].ensemble[0],
    #                                 model.tau)
    #         if j + 1 < grid_c:
    #             local2local_connect(model.ea_oscillator[i][j].ensemble[0],
    #                                 model.ea_oscillator[i][j + 1].ensemble[0],
    #                                 model.tau)
    # </editor-fold>

    # local connection
    ea_Jconnector = []

    for i in range(0, grid_r):
        for j in range(0, grid_c):
            if i + 1 < grid_r:
                J = Jij_connector(ea_oscillator[i][j],
                                        ea_oscillator[i + 1][j],
                                        tau, syn)
                ea_oscillator[i][j].addconnection(J.jensemble[2],
                                                        ea_oscillator[i + 1][j].ensemble[0])
                ea_oscillator[i + 1][j].addconnection(J.jensemble[2],
                                                            ea_oscillator[i][j].ensemble[0])
                ea_Jconnector.append(J)
                # print("connector between (",i,j, ") and (",i+1,j,") constructed")
                # nengo.Connection(J.jensemble[2], ea_oscillator[i][j].NJensemble, synapse=syn)
                # nengo.Connection(J.jensemble[2], ea_oscillator[i][j].NJensemble, synapse=syn)
            if j + 1 < grid_c:
                J = Jij_connector(ea_oscillator[i][j],
                                  ea_oscillator[i][j + 1],
                                  tau, syn)
                ea_oscillator[i][j].addconnection(J.jensemble[2],
                                                        ea_oscillator[i][j + 1].ensemble[0])
                ea_oscillator[i][j + 1].addconnection(J.jensemble[2],
                                                            ea_oscillator[i][j].ensemble[0])
                ea_Jconnector.append(J)
                # print("connector between (", i, j, ") and (", i, j+1, ") constructed")


    # print("total number of Jconnector= " + str(len(ea_Jconnector)))
    # for i in ea_Jconnector:
    #     print(str(J.label))

#probes
    nj_pr = nengo.Probe(ea_oscillator[0][0].test, synapse=0.01)
    h_pr = nengo.Probe(ea_oscillator[0][0].h, synapse=0.01)
    inhibitor_probe = nengo.Probe(inh.ensemble, synapse= 0.01)
    oscillator_probes = [[0 for x in range(grid_c)] for x in range(grid_r)]
    for i in range(grid_r):
        for j in range(grid_c):
            oscillator_probes[i][j] = nengo.Probe(ea_oscillator[i][j].ensemble[0], synapse=0.01)


with nengo.Simulator(model) as sim:
    sim.run(runtime)
    t = sim.trange()
    plt.figure()
    plt.subplot(grid_r+1,1,1)
    plt.plot(t, sim.data[inhibitor_probe], label= "inhibitor", color= 'k')
    plt.legend(prop={'size':13})
    for i in range(grid_r):
        plt.subplot(grid_r + 1, 1, i + 2)
        for j in range(grid_c):
            plt.plot(t, sim.data[oscillator_probes[i][j]],
                     label="oscillator %d" % (j+1))
    plt.legend(prop={'size': 13})
    plt.suptitle('LEGION')
    plt.ylabel('Activities')
    plt.xlabel('Time (sec)')

    plt.figure()
    plt.plot(t,sim.data[h_pr],label="h_pr")
    plt.legend()

    plt.figure()
    plt.plot(t,sim.data[nj_pr],label="nj_pr")
    plt.legend()

    plt.show()
