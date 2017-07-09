import matplotlib.pyplot as plt
import nengo
from datetime import datetime
from func import *
from osc import Oscillator
from inhib import Inhibitor
import pandas as pd
from plotter import plotter
import os
import yagmail
from mem import memory

# "always" show or "ignore" warnings
import warnings
np.seterr(all='warn')
warnings.simplefilter("always")

# adjust value in "const.py"
# if inp is not in "const.py",
# read from the inp created by "createinp.py"
# NOTE: this code is running under python 2.7 because
# we want the code to be compatible with the
# brian package in "createinp.py"
try:
    inp
except:
    txtDataPath = 'Cython_IPEM/txt/' + filename + '_bitmap'
    print('reading inp from' + txtDataPath)
    with open(txtDataPath+'.txt', 'r') as file:
        inp = [[int(digit) for digit in line.split()] for line in file]
        inp = np.asarray(inp)
        grid_r, grid_c = inp.shape
        print('main: inp dimension= ', inp.shape)
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
    for i in range(grid_r):
        for j in range(grid_c):
            data.append(sim.data[oscillator_probes[i][j]][:,0])
            headerstr.append("oscillator %d %d" % (i,j))
            # data = np.concatenate((data, sim.data[oscillator_probes[i][j]]), axis=1)
            # headerstr.append("oscillator %d%d" % (i, j))
    filedir = 'csv/'+filename+'.csv'





    # https://stackoverflow.com/questions/938733/total-memory-used-by-python-process

    pd.DataFrame(np.asarray(data).T).to_csv(filedir, index=False, header=headerstr)


#clear memory
    # del Oscillator
    # del Inhibitor
    # del oscillator_probes
    # del inhibitor_probe

# plot and save png
plotter()
# if os.name == 'nt':
#     plt.show()

# send out picture if using linux
# https://github.com/kootenpv/yagmail/issues/72
if os.name == 'posix':
    subject = 'LEGION '+str(datetime.now())
    body = 'W0= ' + str(W0) + ', W1= ' + str(W1)
    img = 'png/' + filename + '.png'
    yagmail.SMTP('justforthiscode','cnrgntu510').send(
        'theandychung@gmail.com', subject=subject, contents= [body,img])

if df['W0'].shape[0] !=1:
    df.drop(df.index[0], inplace=True)
    df.to_csv('csv/values.csv')
    print df
    execfile('main.py')
# import sys
# # for var in locals().items():
# #     del var
#
# a=0
# for var, obj in locals().items():
#     print var, sys.getsizeof(obj)
#     a=a+sys.getsizeof(obj)
# print a

# make sound after finished
# from music import playmusic
# playmusic()

