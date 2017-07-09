import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from const import dt,runtime
from mem import memory


def plotter():
    # clear all var in memory
    # import sys
    # sys.modules[__name__].__dict__.clear()

    # plot
    try:
        from const import inp
    except ImportError:
        from const import filename
        print('plotter: importing filename from const')
        txtDataPath = 'Cython_IPEM/txt/' + filename + '_bitmap'
        with open(txtDataPath + '.txt', 'r') as file:
            inp = [[int(digit) for digit in line.split()] for line in file]
            inp = np.asarray(inp)
            grid_r, grid_c = inp.shape
            print('plotter: read inp dimension from filename_bitmap.txt: ', inp.shape)
    else:
        print('plotter: filename = test')
        grid_r, grid_c = inp.shape
        filename = 'test'

    # time_data = pd.read_csv(filedir, usecols=['t'])
    # inhib_data = pd.read_csv(filedir, usecols=['inhibitor'])
    #
    # osc_data = pd.read_csv(filedir, usecols=range(2, grid_r * grid_c + 2))
    # for i in range(grid_r - 1, -1, -1):
    #     for j in range(grid_c - 1, -1, -1):
    #         osc_data["oscillator %d%d" % (i, j)] += (5 * i)
    #     inhib_data['inhibitor']+=(grid_r*5)

    filedir = 'csv/' + filename + '.csv'
    reader = pd.read_csv(filedir)
    reader.drop(['t'], axis=1, inplace=True)
    shiftUpVal=  5
    for i in range(grid_r):
        for j in range(grid_c):
            str_theosc = 'oscillator %d %d' % (i, j)
            reader[str_theosc][reader[str_theosc]<-.5] = np.nan
            reader[str_theosc] += (shiftUpVal * (grid_r-i-1))
    reader['inhibitor'][reader['inhibitor'] <= ((reader['inhibitor'].max()-reader['inhibitor'].min())/2)] = np.nan
    reader['inhibitor'] += (grid_r * shiftUpVal)

    ax = reader[reader.columns.difference(['inhibitor'])].plot(legend=False, color='k')
    #using a lot of memory
    ax = reader['inhibitor'].plot(legend=False, color='r')
    ax.yaxis.grid()



    # scale labels
    # scale_y = shiftUpVal
    # scale_x = dt
    # ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))
    # ax.yaxis.set_major_formatter(ticks_y)
    # ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * scale_x))
    # ax.xaxis.set_major_formatter(ticks_x)

    ax.set_xlabel('Time (sec.)')
    ax.set_ylabel('Channels')

    fig = ax.get_figure()
    # fig.gca().get_lines()[0].set_color("red")

    fig.savefig('png/'+filename+'.png')

if __name__ == '__main__':
    plotter()
    plt.show()
