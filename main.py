import matplotlib.pyplot as plt
import csv
import nengo
import datetime
from func import *
from osc import Oscillator
from inhib import Inhibitor
import pandas as pd
# plt.close('all')

# if inp is not in const.py,
# read from the inp created by createinp.py
try:
    inp
except:
    txtDataPath = 'Cython_IPEM/txt/' + filename + '_bitmap'
    print('reading inp from' + txtDataPath)
    with open(txtDataPath+'.txt', 'r') as file:
        inp = [[int(digit) for digit in line.split()] for line in file]
        inp = np.asarray(inp)
        grid_r, grid_c = inp.shape
        print('inp dimension= ', inp.shape)

else:
    grid_r, grid_c = inp.shape
    filename = 'test'

model = nengo.Network(label="hahaha")
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

    #local connections
    for i in range(grid_r):
        for j in range(grid_c):
            if i - 1 >= 0:
                ea_oscillator[i][j].local_connect(ea_oscillator[i - 1][j].ensemble[0],tau)
            if i + 1 < grid_r:
                ea_oscillator[i][j].local_connect(ea_oscillator[i + 1][j].ensemble[0],tau)
            if j - 1 >= 0:
                ea_oscillator[i][j].local_connect(ea_oscillator[i][j - 1].ensemble[0],tau)
            if j + 1 < grid_c:
                ea_oscillator[i][j].local_connect(ea_oscillator[i][j + 1].ensemble[0],tau)

#probes
    inhibitor_probe = nengo.Probe(inh.ensemble, synapse= 0.01)
    oscillator_probes = [[0 for x in range(grid_c)] for x in range(grid_r)]
    for i in range(grid_r):
        for j in range(grid_c):
            oscillator_probes[i][j] = nengo.Probe(ea_oscillator[i][j].ensemble[0], synapse=0.01)


with nengo.Simulator(model) as sim:
    sim.run(runtime)
    t = sim.trange()
    inhib_data = sim.data[inhibitor_probe]
    # sim.data[oscillator_probes[i][j]]

    haederstr = ('t,inhib_data')
    filedir = 'csv_input/'+filename+'.csv'
    # osc_data=[]
    # for i in range(grid_r):
    #     for j in range(grid_c):
    #         osc_data.append(sim.data[oscillator_probes[i][j]])

    # df = pd.DataFrame(osc_data)
    # df.to_csv(filedir, index=False, header=False)
    np.savetxt(filedir,zip(t, sim.data[inhibitor_probe]),header=haederstr, comments="" , delimiter=',')
    # with open(filedir, 'a') as csvfile:
    #     writer = csv.writer(csvfile)

    #     writer.writerow(inhib_data)

    plt.figure()
    plt.subplot(grid_r+1,1,1)
    plt.plot(t,inhib_data , label= "inhibitor", color= 'k')
    plt.title('LEGION')
    plt.legend(prop={'size': 10})
    for i in range(grid_r):
        plt.subplot(grid_r + 1, 1, i + 2)
        for j in range(grid_c):
            plt.plot(t, sim.data[oscillator_probes[i][j]],
                     label="oscillator %d" % (j+1))
    plt.legend(prop={'size': 10})
    plt.suptitle(str(datetime.datetime.now()))
    plt.ylabel('Activities')
    plt.xlabel('Time (sec)')

    plt.show()
