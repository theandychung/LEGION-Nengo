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
    # t=t.reshape((t.shape[0],1))
    # print t.shape
    data = []
    headerstr=[]
    data.append(t)
    headerstr.append('t')
    data.append(sim.data[inhibitor_probe][:,0])
    headerstr.append('inhibitor')
    # data = np.concatenate((t, sim.data[inhibitor_probe]), axis=1)
    # osc_data = [[0 for x in range(grid_c)] for x in range(grid_r)]
    for i in range(grid_r):
        for j in range(grid_c):
            data.append(sim.data[oscillator_probes[i][j]][:,0])
            headerstr.append("oscillator %d%d" % (i,j))
            # data = np.concatenate((data, sim.data[oscillator_probes[i][j]]), axis=1)
            # headerstr.append("oscillator %d%d" % (i, j))


    filedir = 'csv_input/'+filename+'.csv'
    my_df = pd.DataFrame(np.asarray(data).T)
    my_df.to_csv(filedir, index=False, header=headerstr)



# def plotter(filename):



    # plt.figure()
    # plt.subplot(grid_r+1,1,1)
    # plt.plot(time_data,osc_data)
    # plt.title('LEGION')
    # plt.legend(prop={'size': 10})



    # for i in range(grid_r-1,-1,-1):
    #     for j in range(grid_c-1,-1,-1):
    #         plt.plot(t, sim.data[oscillator_probes[i][j]]+i*5,
    #                  label="oscillator %d%d" % (i,j))
    # plt.plot(t, sim.data[inhibitor_probe]+grid_r*5, label="inhibitor", color='k')
    # plt.legend(prop={'size': 10})
    # # plt.suptitle(str(datetime.datetime.now()))
    # plt.ylabel('Activities')
    # plt.xlabel('Time (sec)')

    # plt.show()
