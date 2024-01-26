#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:28:54 2024

@author: frichard
"""

from afbf import process
from numpy.random import seed
from numpy import zeros, std, arange, power, mean, maximum, log, array
from numpy import concatenate, ones, infty, sqrt
from scipy.optimize import lsq_linear
from varprox import Minimize, Varprox_Param
from varprox.models.model_mfbm import Ffun, DFfun
from matplotlib import pyplot as plt


# Experiment parameters
N = 1000  # Size of the observed process.

scales = arange(1, 5)
w_size = 40
w_step = 1


def Simulate_MFBM(H, seed_n=None):
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
    if seed_n is None:
        seed()
    N = H.size

    fbm = process()
    y = zeros((N,))
    for j in range(N):
        seed(seed_n)
        fbm.param = H[j]
        fbm.Simulate(N)
        y[j] = fbm.y[j] / std(fbm.y)

    return y


def Estimate_LocalSemiVariograms(y, scales, w_size, w_step):
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

    Returns
    -------
    v : ndarray of dimension 2
        Semi-variograms at each position (row) and scale (column).

    """

    N = y.size
    Nr = N - max(max(scales), w_size)

    v = zeros((Nr // w_step, scales.size))
    for j in range(scales.size):
        # Increments
        scale = scales[j]
        increm = power(y[:-scale] - y[scale:], 2)
        # Local semi-variogram.
        w_ind = 0
        for w in range(0, Nr, w_step):
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

    H = zeros((N, 1))
    c = zeros((N, 1))
    for j in range(N):
        pb = lsq_linear(X, v[j, :], bounds=(lb, ub))
        H[j] = pb.x[0]
        c[j] = pb.x[1]

    return H, c


# Simulate a Hurst function.
H1 = 0.1
H2 = 0.9
T = arange(N)
N = N - 1
T = sqrt(T / N)
H = (1 - T) * H1 + T * H2
# Simulate a mfbm of Hurst function H.
seed_n = 1
y = Simulate_MFBM(H, seed_n)
# Estimate the local semi-variogram of y.
v = Estimate_LocalSemiVariograms(y, scales, w_size, w_step)
# Estimate the Hurst function.
Hest, c = Estimate_HurstFunction(scales, v)
# Comparison with ground truth.
ind0 = w_size // 2
H = H[ind0:ind0 + Hest.size]
DH = H - Hest
print(mean(DH), std(DH))

plt.figure(1)
plt.plot(H, label="True")
plt.plot(Hest, label="Estimate")
plt.show()

logscales = log(scales)
scales2 = power(scales, 2)

Hest2 = 0.5 * ones(H.shape)
y = ones(H.shape)
w = v.reshape((v.size,))
pb = Minimize(Hest2, y, w, Ffun, DFfun, (0, 1), (0, infty),
              scales2, logscales, 0)
optim_param = Varprox_Param(1e-3, 100, True)
Hest2, c2 = pb.argmin_h(optim_param)
