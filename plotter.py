import matplotlib.pyplot as plt
import pandas as pd
from const import *
import numpy as np
# grid_r = 4
# grid_c = 4
#

def plotter(filename, grid_r, grid_c):
    # time_data = pd.read_csv(filedir, usecols=['t'])
    # inhib_data = pd.read_csv(filedir, usecols=['inhibitor'])
    #
    # osc_data = pd.read_csv(filedir, usecols=range(2, grid_r * grid_c + 2))
    # for i in range(grid_r - 1, -1, -1):
    #     for j in range(grid_c - 1, -1, -1):
    #         osc_data["oscillator %d%d" % (i, j)] += (5 * i)
    #     inhib_data['inhibitor']+=(grid_r*5)
    filedir = 'csv_input/' + filename + '.csv'
    reader = pd.read_csv(filedir)
    del reader["t"]
    for i in range(grid_r - 1, -1, -1):
        for j in range(grid_c - 1, -1, -1):
            reader["oscillator %d%d" % (i, j)] += (5 * i)
    reader['inhibitor'] += (grid_r * 5)
    reader.plot()
    plt.legend(prop={'size': 5})
    plt.show()


if __name__ == '__main__':
    try:
        inp
    except:
        print('importing filename from const')
        txtDataPath = 'Cython_IPEM/txt/' + filename + '_bitmap'
        with open(txtDataPath + '.txt', 'r') as file:
            inp = [[int(digit) for digit in line.split()] for line in file]
            inp = np.asarray(inp)
            grid_r, grid_c = inp.shape
            print('read inp dimension from filename_bitmap.txt: ', inp.shape)
    else:
        print('filename = test')
        grid_r, grid_c = inp.shape
        filename = 'test'
    plotter(filename,grid_r, grid_c)

