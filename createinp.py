import numpy as np
from Cython_IPEM.IPEM import IPEM
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size / 2:]


def plotdismatrix(inp,name):
    Rows, Cols = inp.shape
    inMinY = np.min(inp)
    inMaxY = np.max(inp)
    Scale = abs(inMaxY - inMinY)

    plt.figure()
    for i in range(Rows):
        # for i in range(28,35):
        if (Scale != 0):
            plt.title(name)
            plt.plot((i + 0.05) + 0.9 * (inp[i][:] - inMinY), color="blue")
        else:
            plt.title(name)
            plt.plot((i + 0.05) + 0.9 * (inp[i][:] - inMinY), color="blue")

def call_ipem(filename):
    inInputFileName = filename + '.wav'
    inInputFilePath = 'Cython_IPEM/wav/'
    inOutputFileName = filename + '.txt'
    inOutputFilePath = 'Cython_IPEM/txt/'
    myIPEM = IPEM(inInputFileName, inInputFilePath, inOutputFileName, inOutputFilePath)
    myIPEM.ipem()
    myIPEM.plot()

def autocorr_file(filename):
    dataPath = 'Cython_IPEM/txt/' + filename + '.txt'
    with open(dataPath, 'r') as file:
        inp = [[float(digit) for digit in line.split()] for line in file]
        inp = np.asarray(inp)
        inp = np.transpose(inp)
        # [r,c]= inp.shape
        # print('createinp/autocorr_file: transform inp to',inp.shape)

        # take N point data from music matrix
        datalength = 20
        inp = inp[:,50-datalength/2:50+datalength/2]

        inp = np.apply_along_axis(autocorr, axis=1, arr=inp)
        # inp = inp
        # [R,C]=inp.shape
    np.savetxt('Cython_IPEM/txt/'+filename+'_autocorr.txt', inp, fmt='%.10f')#, comments="" , delimiter=','
    plotdismatrix(inp, filename)

def mapper(filename, threshold= None):
    txtDataPath = 'Cython_IPEM/txt/' + filename + '_autocorr'
    pngDataPath = 'Cython_IPEM/png/' + filename
    with open(txtDataPath+'.txt', 'r') as file:
        inp = [[float(digit) for digit in line.split()] for line in file]
        inp = np.asarray(inp)
        [r, c] = inp.shape
        inp = inp#[:,0:40]
    plt.imsave(pngDataPath + '_grey.png', inp, cmap=cm.gray)
    if threshold == None:
        threshold = inp.mean()
        print('threshold= '+ str(threshold))
    lowValY = 5
    low_values_indices = inp < threshold  # Where values are low
    high_values_indices = inp >= threshold
    inp[low_values_indices] = 0  # All low values set to 0
    inp[high_values_indices] = 1  # All high values set to 1
    plt.imsave(pngDataPath + '.png', inp, cmap=cm.gray)
    np.savetxt('Cython_IPEM/txt/' + filename + '_bitmap.txt', inp, fmt='%i')
    # plotdismatrix(inp, filename + '0&1')


filename = 'bee2'
threshold = None

# call_ipem(filename)
autocorr_file(filename)
mapper(filename, threshold)

plt.show()
