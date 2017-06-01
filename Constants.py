import numpy as np

I =.5
epsilon=.2
gamma=9.0
beta=0.1

# epsilon = .02;
# gamma = 6.0;
# beta = 0.1;

W0 = .1 #weight for local connection (temp)

inp = np.array([[1,0,0],
                [0,0,0],
                [0,0,1]])
# inp = np.array([[1,0,0,1]])
rho = .02 #amplitude of gaussian noise
phi = 4.0 #the rate at which the inhibitor reacts to the stimulation.
W1 = 12.0
W2 = 0.1
theta = 0.05 #x is enabled? if x>theta_sp, h(x)=1;else h(x)=0;
theta_x = -.2
theta_1 = 0.5
theta_z = -.5
kappa = 50
t_th = 7 #for <x>
num_z = 2  # global inhibitor
eta = 10 #control speed of DJ
W_T = 6 #DJ weights
sigma_t = 8.0
sigma_f = 5.0

# grid_r,grid_c = inp.shape
# N_t = grid_c
# theta_2 = 1/(2*N_t)