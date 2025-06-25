#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:28:54 2024

@author: frichard
"""

from afbf import process
from numpy.random import seed
from numpy import zeros, std, arange, power, mean, maximum, minimum, log, array
from numpy import concatenate, ones, infty, sqrt
from scipy.optimize import lsq_linear
from varprox.models.model_mfbm import Ffun_v, DFfun_v
from matplotlib import pyplot as plt
from varprox import Parameters, Minimize

# Optimisation parameters
maxit = 5000
gtol_h = 10e-4

# Experiment parameters
N = 400  # Size of the observed process.

order = 0
scales = arange(1, 5)
w_size = 100  # 990  # 40
w_step = 1  # 990  # 1

H1, H2 = 0.2, 0.2


def Simulate_MFBM(H, seed_n=1):
    """Simulate of multi-fractional Brownian motion of Hurst function H.

    Parameters
    ----------
    H : ndarray
        The hurst function of the process.
    seed_n : int
        A seed number.

    Returns
    -------
    mfbm : ndarray
        A simulation of the process.

    """
    N = H.size

    fbm = process()
    y = zeros((N,))
    for j in range(N):
        seed(seed_n)
        fbm.param = H[j]
        fbm.Simulate(N)
        y[j] = 10 * fbm.y[j] / std(fbm.y)

    return y


def Estimate_LocalSemiVariograms(y, scales, w_size, w_step, order=0):
    """Compute the local semi-variogram of the process.

    Parameters
    ----------
    y : ndarray
        A process.
    scales : ndarray
        Scales at which the semi-variogram is computed.
    w_size : int
        Size of the window where the semi-variogram is computed.
    w_step : int
        Step between two successive positions of computations.
    order: int
        Order of the increments. The default is 0.

    Returns
    -------
    v : ndarray of dimension 2
        Semi-variograms at each position (row) and scale (column).

    """

    N = y.size
    order += 1
    Nr = N - order * max(scales) - w_size

    if Nr < 0:
        raise "Decrease max of scales or w_size."

    v = zeros((Nr // w_step + 1, scales.size))
    for j in range(scales.size):
        # Increments
        scale = scales[j]
        increm = zeros(y.shape)
        increm[:] = y[:]
        for o in range(order):
            increm = increm[:-scale] - increm[scale:]
        increm = power(increm[:-scale] - increm[scale:], 2)
        # Local semi-variogram.
        w_ind = 0
        for w in range(0, Nr, w_step):
            # v[w_ind, j] = 0.5 * power(scale, 2 * 0.7)
            v[w_ind, j] = 0.5 * mean(increm[w:w + w_size])
            w_ind += 1

    return(v)


def Estimate_HurstFunction(scales, v):
    """Estimate the Hurst function using linear regressions.

    Parameters
    ----------
    scales : ndarray
        scales at which the semi-variogram was estimated.
    v : ndarray
        estimes of the semi-variogram.

    Returns
    -------
    c : ndarray
        Variance factors.
    H : ndarray
        Hurst function.

    """
    N = v.shape[0]
    P = v.shape[1]
    v = log(maximum(v, 1e-90))
    scales = 2 * log(scales).reshape((P, 1))
    X = concatenate((scales, ones((P, 1))), axis=1)
    lb = array([0, - infty])
    ub = array([1, infty])

    H = zeros((N,))
    c = zeros((N,))
    for j in range(N):
        pb = lsq_linear(X, v[j, :], bounds=(lb, ub))
        H[j] = pb.x[0]
        c[j] = pb.x[1]

    return H, c


# Simulate a Hurst function.
T = arange(stop=N, step=2)
N = N - 1
T = sqrt(T / N)
H = (1 - T) * H1 + T * H2
H = concatenate((H[::-1], H[1:]))
# H = 0.8 * ones(H.shape)
# Simulate a mfbm of Hurst function H.
seed_n = 1
y = Simulate_MFBM(H, seed_n)
# Estimate the local semi-variogram of y.
v = Estimate_LocalSemiVariograms(y, scales, w_size, w_step, order)

# Estimate the Hurst function by linear regression.
Hest1, c1 = Estimate_HurstFunction(scales, v)

# Comparison with ground truth.
ind0 = (H.size - Hest1.size) // 2
H = H[ind0:ind0 + Hest1.size]
DH = H - Hest1

# Estimation of the Hurst function by minimisation.
scales = scales / (2 * max(scales))
scales2 = power(scales, 2)
logscales = log(scales2)


param_opti = Parameters()
param_opti.load("param_optim.ini")

param_opti.reg.name = None
param_opti.itermax_neg = 500
param_opti.reg.order = 2
param_opti.reg.name = None

x0 = 0.5 * ones(H.shape)
# x0[:] = Hest1[:]

eps = 10e-8
x0 = minimum(maximum(param_opti.bounds_x[0] + eps, x0),
             param_opti.bounds_x[1] - eps)
w = v.reshape((v.size,), order="F")

pb = Minimize(x0, w, Ffun_v, DFfun_v, scales2, logscales, 0)
# param_opti.alpha = pb.h_value() * 1
pb.params = param_opti


Hest2, c2 = pb.argmin_h()


# Regularization parameter x
pb.x[:] = x0[:]
# Weight for y regularization
pb.param.reg.name = 'tv-1d'
pb.param.reg_weight = pb.h_value() * 1
Hest3, c3 = pb.argmin_h()


plt.figure(1)
plt.plot(H, label="Ground truth")
plt.plot(Hest1, label="Linear regression")
plt.plot(Hest2, label="Varpro")
# plt.plot(Hest3, label="Varprox")
plt.legend()
plt.show()
