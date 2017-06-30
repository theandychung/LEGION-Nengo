import numpy as np


runtime = 20
I = 0
epsilon=.2
gamma=9.0
beta=0.1

# epsilon = .02;
# gamma = 6.0;
# beta = 0.1;

W0 = .1 #weight for local connection (temp)

inp = np.array([[1,0,0,0],
                [1,0,0,0],
                [0,0,0,1],
                [0,0,0,1]])

# inp = np.array([[1,1,0,0]])
rho = 0.02 #amplitude of gaussian noise
phi = 3.0 #the rate at which the inhibitor reacts to the stimulation.
W1 = 4.5
W_T = 50 #DJ weights
t_th = .05 #for <x>
W2 = 0.1
theta = 0.05 #x is enabled? if x>theta_sp, h(x)=1;else h(x)=0;
theta_x = -.2
theta_1 = 0.1
theta_z = 0.1
kappa = 50
num_z = 2  # global inhibitor
eta = 10 #control speed of DJ
sigma_t = 8.0
sigma_f = 5.0

c = 1E-10
dt = 0.001
grid_r,grid_c = inp.shape
# N_t = grid_c
# theta_2 = 1/(2*N_t)

default_tau = 3
default_syn = default_tau