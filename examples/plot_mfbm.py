#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimation of the Hurst function of an multifractional Brownian motion.

@author: Frédéric Richard, 2024.
"""

from afbf import process
from numpy.random import seed
from numpy import zeros, std, arange, power, mean, maximum, minimum, log, array
from numpy import concatenate, ones, infty, sqrt
from scipy.optimize import lsq_linear
from varprox import Minimize
from varprox._minimize import tv
from varprox.models.model_mfbm import Ffun, DFfun_v
from matplotlib import pyplot as plt

# Experiment parameters
N = 1000  # Size of the observed process.

order = 0
scales = arange(1, 5)
w_size = 40
w_step = 1


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
H1 = 0.1
H2 = 0.9
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

# Estimate the Hurst function by minimisation.
scales = scales / (2 * max(scales))
scales2 = power(scales, 2)
logscales = log(scales2)

Hest2 = 0.5 * ones(H.shape)
Hest2[:] = Hest1[:]
Hest2 = minimum(maximum(0.0001, Hest2), 0.9999)
w = v.reshape((v.size,), order="F")
pb = Minimize(Hest2, w, Ffun, DFfun_v, scales2, logscales, 0)
# pb.param.load("plot_mfbm.ini")

pb.param.bounds_x = (0.0001, 0.9999)
pb.param.bounds_y = (0, infty)
Hest2, c2 = pb.argmin_h()


# Estimate of the Hurst function by minimisation with a penalization.
# Regularization parameter x.
pb.param.reg.weight = pb.h_value() / tv(pb.x) * pb.K * 10e20
# Weight for y regularization.
pb.param.reg.name = 'tv-1d'
Hest3, c3 = pb.argmin_h()


plt.figure(1)
plt.plot(H, label="Ground truth")
plt.plot(Hest1, label="Linear regression")
plt.plot(Hest2, label="Varpro")
plt.plot(Hest3, label="Varprox")
plt.legend()
plt.show()
