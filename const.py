import numpy as np
from nengo.utils.functions import piecewise

#total simulation time
runtime = .1

#input
##for testing oscillator
I = 0
##for LEGION
# inp = np.array([[1,1,0,0]])

# inp = np.array([[1,1,0,0],
#                 [0,0,0,0],
#                 [0,0,0,1]])

inp = np.array([[1,1,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,1]])

#if no inp is defined,
#read input from "filename".txt file
filename = 'bee2'



#parameter values
##handly ones
###weight for local connection
W0 = 3
###weight of inhibition
W1 = 4#6.5

##better not touch this
epsilon=.2 #osc
gamma=9.0 #osc
beta=0.1 #osc
rho = 0.02 #osc-amplitude of gaussian noise

phi = 3.0 #the rate at which the inhibitor reacts to the stimulation.

W_T = 54#40 #DJ weights
t_th = .05 #for <x>
theta = 0.05 #x is enabled? if x>theta_sp, h(x)=1;else h(x)=0;
theta_x = -.2
theta_1 = 0.1
theta_z = 0.1
kappa = 50
eta = 10 #control speed of DJ
sigma_t = 8.0
sigma_f = 5.0

c = 1E-10
dt = 0.001

default_tau = 3
default_syn = default_tau



