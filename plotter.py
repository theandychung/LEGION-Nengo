import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from const import dt
from mem import memory


def plotter(colar=None, marks=True, **kwargs):
    # colar: oscillator colors.
    #   None: generate random colors
    # marks: draw only the tips

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
            if marks==True:
                reader[str_theosc][reader[str_theosc]<-.5] = np.nan
            reader[str_theosc] += (shiftUpVal * (grid_r-i-1))
    if marks==True:
        reader['inhibitor'][reader['inhibitor'] <= ((reader['inhibitor'].max()-reader['inhibitor'].min())/2)] = np.nan
    reader['inhibitor'] += (grid_r * shiftUpVal)

    # generate colors if colar=None
    if colar == None:
        from gen_color import generate_new_color
        colar=[]
        for i in range(0, grid_c):
            colar.append(generate_new_color(colar, pastel_factor=.1))
            # colar = colar

    # plot
    fig = plt.figure()
    ax = reader[reader.columns.difference(['inhibitor'])].plot(legend=False, color= colar*grid_r, ax=fig.gca())
    reader['inhibitor'].plot(legend=False, color='r', ax=ax)
    ax.yaxis.grid()

    # scale labels
    scale_y = shiftUpVal
    scale_x = dt
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))
    ax.yaxis.set_major_formatter(ticks_y)
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * scale_x))
    ax.xaxis.set_major_formatter(ticks_x)
    ax.set_xlabel('Time (sec.)')
    ax.set_ylabel('Channels')

    # adjust legends
    plt.legend(loc='center left',
               bbox_to_anchor=(1.0, 0.5))
               # fontsize = 'xx-large',
               # ncol = 2,
               # handleheight = 2.4,
               # labelspacing = 0.01
    fig.subplots_adjust(right=0.75) #move plot for legend

    # save picture
    # fig = ax.get_figure()
    # fig.gca().get_lines()[0].set_color("red")
    fig.savefig('png/'+filename+'.png')

if __name__ == '__main__':

    plotter(colar='k', marks=False)
    plt.show()
