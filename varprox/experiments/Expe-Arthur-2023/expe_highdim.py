# -*- coding: utf-8 -*-
r"""
Experiments of fitting variogram of an anisotropic fractional Brownian field.
"""
from matplotlib import pyplot as plt
import time
import numpy as np
from afbf import coordinates, perfunction, tbfield, process
from afbf.Simulation.TurningBands import tbparameters
from varprox.models.model_afbf_2 import FitVariogram, FitVariogram_ADMM
from varprox.models.model_afbf_2 import Fit_Param
from numpy.random import default_rng, rand
import pickle

rng = default_rng()

# Experiment parameters
Nbexpe = 1  # Number of experiments.
Tvario = False  # True if the the theoretical semi-variogram is fitted.
noise = 1  # 1 if model with noise and 0 otherwise.
display = False  # Display results of each experiment.
save = False  # Save the results.
stepK = 1  # Subsampling factor for selecting turning bands.
K = 128  # Number of parameters to estimate with varpro.
alpha = 10  # Weight for tau regularization.

# Optimisation parameters.
multigrid = True  # If true, use a multigrid approach.
maxit = 5000  # Maximal number of iterations.
gtol = 0.001  # tolerance.
verbose = True  # set to 1 to visualize.

myparam = Fit_Param(noise, None, multigrid, maxit, gtol, verbose)


# Model parameters.
N = 40  # size of the grid for the definition of the semi-variogram.
step = 2
M = 512  # size of field realization.
# Number of parameters for the Hurst function.
J = 500

# Definition of turning-band parameters.
tb = tbparameters(J)
kangle = tb.Kangle[0::stepK]
fintermid = kangle[0:-1]
finter = (kangle[1:] + kangle[0:-1]) / 2
J = finter.size

# Definition of the reference model.
topo = perfunction('step', finter.size)
hurst = perfunction('step', finter.size)
topo.finter[0, :] = finter[:]
hurst.finter[0, :] = finter[:]
model = tbfield('reference', topo, hurst, tb)

# Definition of a fbm to sample the Hurst function.
fbm = process()
fbm.param = 0.9

# Definition of the estimated model (using varprox).
etopo = perfunction('step', finter.size)
ehurst = perfunction('step', finter.size)
etopo.finter[0, :] = finter[:]
ehurst.finter[0, :] = finter[:]
emodel1 = tbfield('reference', etopo, ehurst, tb)

# Definition of the initial estimated model (using varpro).
topo0 = perfunction('step', K)
hurst0 = perfunction('step', K)
finter0 = np.linspace(- np.pi / 2, np.pi / 2, K + 1, True)[1:]
topo0.finter[0, :] = finter0[:]
hurst0.finter[0, :] = finter0[:]
model0 = tbfield('reference', topo0, hurst0)

# Grid where to simulate images.
if not Tvario:
    coord = coordinates(M)
    coord.N = N

# Lags where to compute the semi-variogram.
lags = coordinates()
lags.DefineSparseSemiBall(N)
sc = np.sqrt(np.power(lags.xy[:, 0], 2) + np.power(lags.xy[:, 1], 2))


Tau0 = np.zeros((Nbexpe, J))
Beta0 = np.zeros((Nbexpe, J))
Tau1 = np.zeros((Nbexpe, J))
Beta1 = np.zeros((Nbexpe, J))
Tau2 = np.zeros((Nbexpe, J))
Beta2 = np.zeros((Nbexpe, J))
err_fit = err_est = err_tes = 0
time_c1 = 0
time_c2 = 0
for expe in range(Nbexpe):
    # Change model parameters.
    np.random.seed(expe)
    fbm.Simulate(J)
    fparam = fbm.y[:, 0]

    fparam = fparam + np.flip(fparam)
    fmin = np.min(fparam)
    fmax = np.max(fparam)
    fext = np.random.rand() * 0.9
    flow = 0.05 + np.random.rand() * (0.9 - fext)

    if fmin != fmax:
        fparam = flow + fext * (fparam - fmin) / (fmax - fmin)
    else:
        fparam = flow * np.ones(fparam.shape)
    model.hurst.fparam[0, :] = fparam[:]
    model.NormalizeModel()
    print("Topo tv-norm:", np.mean(np.absolute(np.diff(
        model.topo.fparam[0, 0:-2]))))
    print("Hurst tv-norm:", np.mean(np.absolute(np.diff(
        model.hurst.fparam[0, 0:-2]))))

    Tau0[expe, :] = topo.fparam[0, :]
    Beta0[expe, :] = hurst.fparam[0, :]

    # Computation of the theoretical semi-variogram.
    model.ComputeApproximateSemiVariogram(lags)
    w0 = np.zeros(model.svario.values.size)
    w0[:] = model.svario.values[:, 0]

    if noise == 1:
        # Noise variance.
        s0 = np.random.rand() * np.min(w0)
    else:
        s0 = 0

    # Semivariogram to be fitted (theoretical / empirical, noise / no noise).
    if Tvario:
        w = np.zeros(w0.shape)
        w[:] = w0[:]
        if noise == 1:
            w = w + s0
    else:
        # Simulate a field realization.
        z = model.Simulate(coord)
        if noise == 1:
            z.values = z.values +\
                np.sqrt(s0) * rng.standard_normal(z.values.shape)

        # Compute the empirical semi-variogram.
        evario = z.ComputeEmpiricalSemiVariogram(lags)
        w = evario.values[:, 0]
        # Evaluate the estimation error.
        err_est += np.mean(np.abs(w0 + s0 - w))

    # Initial variogram fitting with varpro.
    t0 = time.perf_counter()
    emodel0, wt = FitVariogram(model0, lags, w, myparam, alpha)
    t1 = time.perf_counter()
    time_c1 += t1 - t0

    emodel0.topo.Evaluate(fintermid)
    emodel0.hurst.Evaluate(fintermid)
    emodel1.topo.fparam[0, :] = emodel0.topo.values[0, :]
    emodel1.hurst.fparam[0, :] = emodel0.hurst.values[0, :]
    Tau1[expe, :] = emodel1.topo.fparam[0, :]
    Beta1[expe, :] = emodel1.hurst.fparam[0, :]

    # Variogram fitting with varprox.
    t0 = time.perf_counter()
    emodel2, w1 = FitVariogram_ADMM(emodel1, lags, w0, myparam)
    # emodel2, w1 = FitVariogram(emodel1, lags, w0, myparam, alpha)
    t1 = time.perf_counter()
    time_c2 += t1 - t0
    Tau2[expe, :] = emodel2.topo.fparam[0, :]
    Beta2[expe, :] = emodel2.hurst.fparam[0, :]

    print('expe %d / %d.' % (expe + 1, Nbexpe))


print('Experiment report:')
print('Number of coefficients (beta=%d, tau=%d)' % (J, J))
print('Radial precision: %e' % (np.pi / J))
print('Theoretical variogram: %d' % (Tvario))
print("Varpro")
print('Error on coefficients:')
print('L1 : beta=%e, tau=%e' % (np.mean(np.absolute(Beta1 - Beta0), axis=None),
                                np.mean(np.absolute(Tau1 - Tau0), axis=None)))
print('bias: beta=%e, tau=%e' % (np.mean(Beta1 - Beta0, axis=None),
                                 np.mean(Tau1 - Tau0, axis=None)))
print('RMSE: beta=%e, tau=%e' % (
    np.sqrt(np.mean(np.power(Beta1 - Beta0, 2), axis=None)),
    np.sqrt(np.mean(np.power(Tau1 - Tau0, 2), axis=None))))
print('Mean execution time (varpro): %e (sec)' % (time_c1 / Nbexpe))
print("Varprox")
print('Error on coefficients:')
print('L1 : beta=%e, tau=%e' % (np.mean(np.absolute(Beta2 - Beta0), axis=None),
                                np.mean(np.absolute(Tau2 - Tau0), axis=None)))
print('bias: beta=%e, tau=%e' % (np.mean(Beta2 - Beta0, axis=None),
                                 np.mean(Tau2 - Tau0, axis=None)))
print('RMSE: beta=%e, tau=%e' % (
    np.sqrt(np.mean(np.power(Beta2 - Beta0, 2), axis=None)),
    np.sqrt(np.mean(np.power(Tau2 - Tau0, 2), axis=None))))
print('Mean execution time (varprox): %e (sec)' % (time_c2 / Nbexpe))


# if save:
#     with open("results.pickle", "wb") as f:
#         pickle.dump([Beta0, Tau0, Beta1, Tau1,
#                      Tvario, noise,
#                      multigrid, maxit, gtol,
#                      N, step, M,
#                      K, J], f)
