# -*- coding: utf-8 -*-
r"""
Tomography inverse problem example using Varprox.
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from dataclasses import dataclass
from numpy import linalg as LA
from varprox._minimize2D import Minimize2D
from varprox._parameters import Parameters, SolverParam, RegParam
import time
from datetime import datetime
from os.path import isfile
# ============================================================================ #


# ============================================================================ #
#                              FUNCTIONS DEFINITION                            #
# ============================================================================ #
def Ffun(x, s, theta, dim_grid):   
    k = len(theta)
    m = len(s)
    A = np.zeros((m*k, dim_grid*dim_grid))
    kernel = lambda t : np.exp(-500*t**2)
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, dim_grid),
                                 np.linspace(-1, 1, dim_grid))
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    for i in range(k):
        theta_tmp = theta + x[i]
        s_tmp = s + x[i + k]
        for j in range(m):
            A[m*i + j, :] = kernel(grid_x*np.cos(theta_tmp[i]) \
                                   + grid_y*np.sin(theta_tmp[i]) + s_tmp[j])
    return A


def DFfun(x, y, s, theta, dim_grid):    
    k = len(theta)
    n = dim_grid
    m = len(s)
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, n),
                                 np.linspace(-1, 1, n))
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    G = np.zeros((k*m, 2*k))
    dkernel = lambda s : -1000*s*np.exp(-500*s**2)

    for i in range(k):
        theta_tmp = theta + x[i]
        s_tmp = s + x[i + k]
        
        for j in range(m):
            dtheta = dkernel(grid_x*np.cos(theta_tmp[i]) \
                             + grid_y*np.sin(theta_tmp[i]) + s_tmp[j])\
                            *(-grid_x*np.sin(theta_tmp[i]) \
                              + grid_y*np.cos(theta_tmp[i]))
            G[len(s)*i + j, i] = dtheta.dot(y)
            ds = dkernel(grid_x*np.cos(theta_tmp[i]) \
                         + grid_y*np.sin(theta_tmp[i]) + s_tmp[j])
            G[len(s)*i + j, i + k] = ds.dot(y)
    return G


def build_d(A, y, sigma, rng):
    return rng.normal(A@y, sigma)


def plot_data(data, theta, s, k, m):
    fig,ax = plt.subplots(1,1)
    ax.imshow(data.reshape((k,m)).T,extent=(theta[0],theta[-1],s[0],s[-1]))
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$s$')
    ax.set_aspect(2)
    #plt.savefig('observations.png', bbox_inches='tight')
    plt.show()
    return fig, ax


def plot_results(results, n, labels=[]):
    N = len(results)
    fig, ax = plt.subplots(1, N, sharey=True)
    if N > 1:
        for i in range(N):
            x,y = results[i]
            ax[i].imshow(y.reshape((n,n)), extent=(0,1,0,1), cmap='gray',
                         vmin=0, vmax=1)
            if labels:
                ax[i].set_title(labels[i])
    else:
        x,y = results[0]
        ax.imshow(y.reshape((n,n)), extent=(0,1,0,1), cmap='gray',
                  vmin=0, vmax=1)
    #plt.savefig('reconstruction.png', bbox_inches='tight')
    plt.show()
    return fig, ax


def convert_time(time):
    minu, sec = divmod(round(time), 60)
    h, minu = divmod(minu, 60)
    return (h, minu, sec)


def print_time(time):
    (h, minu, sec) = convert_time(time)
    return "{:e} (sec) / {:d}h {:d}min {:d}s".format(time, h, minu, sec)
# ============================================================================ #


# ============================================================================ #
#                                 INITIALIZATION                               #
# ============================================================================ #

# --- Define constants
# Experiment set-up
N = 50  # Dimension of the discretization grid
M = 50  # Offsets sampled regularly in [-1.5 ; 1.5] (number of data points)
K = 50  # Angles sampled regularly in [0 ; 2 Pi]
NOISE_STD = 1  # Standard deviation for the Gaussian noise on the data vector
# Iterative algorithm parameters
MAXIT = 100  # Maximum number of iterations
GTOL = 5E-3  # Tolerance for the stopping criterion
VERBOSE = True  # Is the algorithm verbose?
REG_WEIGHT = 0.01  # Regularization weight for x
ALPHA = 5  # Regularization weight for y

# Number of experiments
Nbexpe = 1
# Display results of each experiment
display = False
# Save the results
save = True

param = Parameters(gtol=GTOL, maxit=MAXIT, verbose=VERBOSE,
                   reg=RegParam("tv-1d", REG_WEIGHT))

# --- Set up the seed for random generation
rng = np.random.default_rng(seed=6)

# --- Create the data
s = np.linspace(-1.5, 1.5, M)  # Offsets
theta = np.linspace(0, 2*np.pi, K)  # Angles

yt = np.zeros((N, N))
yt[10:20,20:40] = 1
yt[30:45,5:20] = 0.8
yt[34:40,35:40] = 1.5
yt = yt.flatten()
xt = np.concatenate((0.05*rng.normal(size=K), 0.05*rng.normal(size=K)))
A = Ffun(xt, s, theta, N)
d = build_d(A, yt, NOISE_STD, rng)

# --- Bounds on variables
param.bounds_x = (-np.inf, np.inf)
param.bounds_y = (0, np.inf)
# ============================================================================ #


# ============================================================================ #
#                    Solve the Variable Projection problem                     #
# ============================================================================ #
# --- Generate an initial point
x0 = np.zeros(xt.shape)
A0 = Ffun(x0, s, theta, N)
tmp = lsq_linear(A0, d, bounds=param.bounds_y)
y0 = tmp.x
# Estimate a solution using Varproj
pb_proj = Minimize2D(x0, d, Ffun, DFfun, s, theta, N)
param.solver_param = SolverParam(1e-4, 5000)
param.alpha = ALPHA
param.reg.name = None
pb_proj.params = param
time_proj_1 = 0
t1_proj = time.perf_counter()
x_proj, y_proj = pb_proj.argmin_h()
t2_proj = time.perf_counter()
time_proj = t2_proj - t1_proj
# Estimate a solution using Varprox
pb_prox = Minimize2D(x0, d, Ffun, DFfun, s, theta, N)
param.solver_param = SolverParam(1e-4, 5000)
param.alpha = ALPHA
param.reg.name = "tv-1d"
param.reg.order = 2
pb_prox.params = param
time_prox_1 = 0
t1_prox = time.perf_counter()
x_prox, y_prox = pb_prox.argmin_h()
t2_prox = time.perf_counter()
time_prox = t2_prox - t1_prox
# ============================================================================ #


# ============================================================================ #
#                                   Save data                                  #
# ============================================================================ #
date = datetime.today().strftime('%Y-%m-%d')
fname = "data_tomo_" + date
if isfile(fname):
    fname = fname + "_bis"
with open(fname, 'wb') as f:
    np.save(f, param)
    np.save(f, d)
    np.save(f, xt)
    np.save(f, x_proj)
    np.save(f, y_proj)
    np.save(f, x_prox)
    np.save(f, y_prox)
# ============================================================================ #


# ============================================================================ #
#                                  Show results                                #
# ============================================================================ #
print("Parameters:")
print("  - Nb of iter         = ", MAXIT)
print("  - Alpha  (reg. in y) = ", ALPHA)
print("  - Lambda (reg. in x) = ", REG_WEIGHT)
print("Errors:")
print("  - y (Varproj) = ", LA.norm(y_proj-yt)/LA.norm(yt))
print("  - x (Varproj) = ", LA.norm(x_proj-xt)/LA.norm(xt))
print("  - y (Varprox) = ", LA.norm(y_prox-yt)/LA.norm(yt))
print("  - x (Varprox) = ", LA.norm(x_prox-xt)/LA.norm(xt))
print("Running times:")
print("  - Varproj = ", print_time(time_proj))
print("  - Varprox = ", print_time(time_prox))
plot_data(d, theta, s, K, M)
plot_results([(xt,yt),(x0,y0),(x_prox,y_prox),(x_proj,y_proj)], N,
             ['ground truth','initial reconstruction','Varproj reconstruction',
              'Varprox reconstruction'])
# ============================================================================ #
