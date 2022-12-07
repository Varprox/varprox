# -*- coding: utf-8 -*-
r"""
Experiments of fitting variogram of an anisotropic fractional Brownian field.
"""
from matplotlib import pyplot as plt
import time
import numpy as np
from afbf import coordinates, perfunction, tbfield
from varprox.models.model_afbf import FitVariogram
from numpy.random import default_rng
import pickle

rng = default_rng()

# Experiment parameters
Nbexpe = 10  # Number of experiments.
Tvario = False  # True if the the theoretical semi-variogram is fitted.
noise = 1  # 1 if model with noise and 0 otherwise.
display = False  # Display results of each experiment.
save = False  # Save the results.

# Optimisation parameters.
multigrid = True  # If true, use a multigrid approach.
maxit = 5000  # Maximal number of iterations.
gtol = 0.001  # tolerance.
verbose = 0  # set to 1 to visualize.

# Model parameters.
N = 40  # size of the grid for the definition of the semi-variogram.
step = 2
M = 256  # size of field realization.
# Number of parameters for the Hurst function.
K = 2**2
J = K  # Number of parameters for the topothesy function.

# Definition of the reference model.
topo = perfunction('step', J)
hurst = perfunction('step', K)
Jinter = topo.finter.size
Kinter = hurst.finter.size
It = np.linspace(- np.pi / 2, np.pi / 2, Jinter + 1, True)[1:]
Ih = np.linspace(- np.pi / 2, np.pi / 2, Kinter + 1, True)[1:]
model = tbfield('reference', topo, hurst)
# Sampling mode for Hurst parameters.
hurst.SetStepSampleMode(mode_cst='unif', a=0.05, b=0.95)
fparam = perfunction('Fourier', 2)

# Grid where to simulate images.
if not Tvario:
    coord = coordinates(M)
    coord.N = N

# Lags where to compute the semi-variogram.
lags = coordinates()
lags.DefineSparseSemiBall(N)
sc = np.sqrt(np.power(lags.xy[:, 0], 2) + np.power(lags.xy[:, 1], 2))


J = topo.fparam.size
K = hurst.fparam.size
Tau0 = np.zeros((Nbexpe, J))
Beta0 = np.zeros((Nbexpe, K))
Tau1 = np.zeros((Nbexpe, J))
Beta1 = np.zeros((Nbexpe, K))
err_fit = err_est = err_tes = 0
time_c = 0
for expe in range(Nbexpe):
    # Change model parameters.
    np.random.seed(expe)

    fparam.SampleFourierCoefficients()
    fparam.Evaluate(Ih)
    fmin = np.min(fparam.values)
    fmax = np.max(fparam.values)
    fext = np.random.rand() * 0.9
    flow = 0.05 + np.random.rand() * (0.9 - fext)
    if (K > 1) and (fmin != fmax):
        fparam.values = flow + fext * (fparam.values - fmin) / (fmax - fmin)
    else:
        fparam.values = flow * np.ones(fparam.values.shape)

    model.hurst.ChangeParameters(fparam.values, Ih)
    model.NormalizeModel()

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

    # Variogram fitting.
    t0 = time.time()
    emodel, wt = FitVariogram(model, lags, w, noise, None,
                              multigrid, maxit, gtol, verbose)

    t1 = time.time()
    time0 = t1 - t0
    time_c += time0

    Tau1[expe, :] = emodel.topo.fparam[0, :]
    Beta1[expe, :] = emodel.hurst.fparam[0, :]

    # Computation of absolute and relative errors on variograms.
    emodel.ComputeApproximateSemiVariogram(lags)
    we = np.zeros(emodel.svario.values.size)
    we[:] = emodel.svario.values[:, 0]

    err_tes += np.mean(np.absolute(we - wt - emodel.noise))
    err_fit0 = np.mean(np.abs(we - w0))
    err_fit += err_fit0

    print('expe %d / %d: error = %e  (time = %e sec)' % (expe + 1,
                                                         Nbexpe,
                                                         err_fit0,
                                                         time0))

    if display:
        print(Tau0[expe, :])
        print(Tau1[expe, :])
        print(Beta0[expe, :])
        print(Beta1[expe, :])
        plt.plot(sc, w0, 'gx', sc, we, 'ro')
        plt.show()

print('Experiment report:')
print('Number of coefficients (beta=%d, tau=%d)' % (J, K))
print('Radial precision: %e' % (np.pi / J))
print('Theoretical variogram: %d' % (Tvario))
if Tvario is False:
    print('Variogram estimation error: %e' % (err_est / Nbexpe))
print('Variogram fitting error: %e' % (err_fit / Nbexpe))
print('Error on coefficients:')
print('L1 : beta=%e, tau=%e' % (np.mean(np.absolute(Beta1 - Beta0), axis=None),
                                np.mean(np.absolute(Tau1 - Tau0), axis=None)))
print('bias: beta=%e, tau=%e' % (np.mean(Beta1 - Beta0, axis=None),
                                 np.mean(Tau1 - Tau0, axis=None)))
print('RMSE: beta=%e, tau=%e' % (
    np.sqrt(np.mean(np.power(Beta1 - Beta0, 2), axis=None)),
    np.sqrt(np.mean(np.power(Tau1 - Tau0, 2), axis=None))))
print('Mean execution time: %e (sec)' % (time_c / Nbexpe))

if save:
    with open("results.pickle", "wb") as f:
        pickle.dump([Beta0, Tau0, Beta1, Tau1,
                     Tvario, noise,
                     multigrid, maxit, gtol,
                     N, step, M,
                     K, J], f)
