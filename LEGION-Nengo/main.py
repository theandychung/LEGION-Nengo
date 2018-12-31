# <editor-fold desc="import...">
""" delay the execution so I can go to sleep"""
from func import *
import matplotlib.pyplot as t
import nengo
from Oscillator import Oscillator
from Inhibitor import Inhibitor
import pandas as pd
from plotter import plotter

"""...to the user
adjust value in "const.py"

if inp is not in "const.py", read the input created by "createinp.py"
"""

try:
    inp
except:
    txtDataPath = Cython_IPEM_Folder+ '/txt/' + filename + '_bitmap'
    print('reading inp from' + txtDataPath)
    with open(txtDataPath+'.txt', 'r') as file:
        inp = [[int(digit) for digit in line.split()] for line in file]
        inp = np.asarray(inp)
        grid_r, grid_c = inp.shape
        print('main: inp dimension= ', inp.shape)
else:
    grid_r, grid_c = inp.shape
    filename = 'test'

model = nengo.Network(label="LEGION in Nengo")
with model:
    model.config[nengo.Ensemble].neuron_type = nengo.Direct()# force direct
    tau = 4
    syn = .2
    inh = Inhibitor(tau, syn)

    # create input nodes, oscillators, and
    # oscillators and inhib connections
    ea_oscillator = [[0 for _ in range(grid_c)] for _ in range(grid_r)]
    for i in range(grid_r):
        for j in range(grid_c):
            ea_oscillator[i][j] = Oscillator(tau, syn, i, j)
            ea_oscillator[i][j].set_input(inp[i][j])
            inh.inhibitorToLocal(ea_oscillator[i][j].ensemble[0])

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
    inhibitor_probe = nengo.Probe(inh.ensemble, synapse=0.01)
    oscillator_probes = [[0 for _ in range(grid_c)] for _ in range(grid_r)]
    for i in range(grid_r):
        for j in range(grid_c):
            oscillator_probes[i][j] = nengo.Probe(ea_oscillator[i][j].ensemble[0], synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(runtime)
    t = sim.trange()
    data = []
    headerstr=[]
    data.append(t)
    headerstr.append('t')
    data.append(sim.data[inhibitor_probe][:,0])
    headerstr.append('inhibitor')
    for i in range(grid_r):
        for j in range(grid_c):
            data.append(sim.data[oscillator_probes[i][j]][:,0])
            headerstr.append("oscillator %d %d" % (i,j))
    filedir = 'csv/'+filename+'.csv'
    pd.DataFrame(np.asarray(data).T).to_csv(filedir, index=False, header=headerstr)

# plot and save
if grid_c > 8 or grid_r > 8:
    plotter(colar='k',marks=True)
else:
    plotter()