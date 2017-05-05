from nengo.utils.functions import piecewise

I =.5
epsilon=.2
gamma=6.0
beta=0.1

# epsilon = .02;
# gamma = 6.0;
# beta = 0.1;

rho = .02; #amplitude of gaussian noise
phi = 3.0; #the rate at which the inhibitor reacts to the stimulation.
W1 = 0.1;
W2 = 0.1;
theta = 0.05; #x is enabled? if x>theta_sp, h(x)=1;else h(x)=0;
theta_x = -.2;
theta_1 = 0.1;
theta_z = 0.1;
kappa = 50;
t_th = 7; #for <x>
num_z = 2;  # global inhibitor
eta = 10; #control speed of DJ
W_T = 6; #DJ weights
sigma_t = 8.0;
sigma_f = 5.0;