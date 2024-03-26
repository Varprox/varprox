# -*- coding: utf-8 -*-
r"""
Experiments of fitting variogram of an anisotropic fractional Brownian field.
"""
import time
import numpy as np
from afbf import coordinates, perfunction, tbfield, process
from varprox.models.model_afbf import FitVariogram
from varprox.ParamsReader import ParamsReader
from numpy.random import default_rng

import pickle

# ============================ Auxiliary functions =========================== #


def print_report(title, beta_est, tau_est, beta_grd, tau_grd, time, nbexpe):
    diff_beta = beta_est - beta_grd
    diff_tau = tau_est - tau_grd
    (h, minu, sec) = convert_time(time)

    print(' ' + title)
    print('    Error on coefficients:')
    print('    L1 : beta={:e}, tau={:e}'.format(
        np.mean(np.absolute(diff_beta), axis=None),
        np.mean(np.absolute(diff_tau), axis=None)))
    print('    bias: beta={:e}, tau={:e}'.format(
        np.mean(diff_beta, axis=None),
        np.mean(diff_tau, axis=None)))
    print('    RMSE: beta={:e}, tau={:e}'.format(
        np.sqrt(np.mean(np.power(diff_beta, 2), axis=None)),
        np.sqrt(np.mean(np.power(diff_tau, 2), axis=None))))
    print('    Mean execution time : {:e} (sec) / {:d}h {:d}min {:d}s'
          .format(time / nbexpe, h, minu, sec))


def convert_time(time):
    minu, sec = divmod(round(time), 60)
    h, minu = divmod(minu, 60)
    return (h, minu, sec)

# ============================================================================ #


# Name of the configuration file containing the parameters
CONFIG_FILE = 'expe_config.ini'

# Read parameters from the configuration file
myreader = ParamsReader(CONFIG_FILE)
myparam = myreader.get_optim_param()
(Nbexpe, Tvario, display, save) = myreader.init_expe_param()
(grid_dim, grid_step, field_size, hurst_dim, _, noise) = myreader.init_model_param()

# Initialization a new random generator
rng = default_rng()

# Definition of turning-band parameters
# tb = tbparameters(J)
# kangle = tb.Kangle[0::stepK]
# fintermid = kangle[0:-1]
# finter = (kangle[1:] + kangle[0:-1]) / 2
# J = finter.size

# Definition of the reference model
topo = perfunction('step', hurst_dim)
hurst = perfunction('step', hurst_dim)
finter = np.linspace(- np.pi / 2, np.pi / 2, hurst.finter.size + 1, True)
fintermid = (finter[0:-1] + finter[1:]) / 2
finter = finter[1:]
model = tbfield('reference', topo, hurst)
tb = model.tb

# topo = perfunction('step', finter.size)
# hurst = perfunction('step', finter.size)
# topo.finter[0, :] = finter[:]
# hurst.finter[0, :] = finter[:]
# model = tbfield('reference', topo, hurst, tb)
# model = tbfield('reference', topo, hurst)

# Definition of a fbm to sample the Hurst function
fbm = process()
fbm.param = 0.5

# Definition of the estimated model (using varprox)
etopo = perfunction('step', finter.size)
ehurst = perfunction('step', finter.size)
etopo.finter[0, :] = finter[:]
ehurst.finter[0, :] = finter[:]
emodel1 = tbfield('reference', etopo, ehurst, tb)
etopo3 = perfunction('step', finter.size)
ehurst3 = perfunction('step', finter.size)
etopo3.finter[0, :] = finter[:]
ehurst3.finter[0, :] = finter[:]
emodel3 = tbfield('reference', etopo3, ehurst3, tb)

# Definition of the initial estimated model (using varproj)
topo0 = perfunction('step', hurst_dim)
hurst0 = perfunction('step', hurst_dim)
finter0 = np.linspace(- np.pi / 2, np.pi / 2, hurst_dim + 1, True)[1:]
topo0.finter[0, :] = finter0[:]
hurst0.finter[0, :] = finter0[:]
model0 = tbfield('reference', topo0, hurst0)

# Grid where to simulate images
if not Tvario:
    coord = coordinates(field_size)
    coord.N = field_size

# Lags where to compute the semi-variogram
lags = coordinates()
lags.DefineSparseSemiBall(grid_dim)
lags.N = field_size
sc = np.sqrt(np.power(lags.xy[:, 0], 2) + np.power(lags.xy[:, 1], 2))

Tau0 = np.zeros((Nbexpe, hurst_dim))
Beta0 = np.zeros((Nbexpe, hurst_dim))
Tau1 = np.zeros((Nbexpe, hurst_dim))
Beta1 = np.zeros((Nbexpe, hurst_dim))
Tau2 = np.zeros((Nbexpe, hurst_dim))
Beta2 = np.zeros((Nbexpe, hurst_dim))
err_fit = err_est = err_tes = 0
time_c1 = 0
time_c2 = 0
for expe in range(Nbexpe):
    # Change model parameters
    np.random.seed(expe)
    fbm.Simulate(hurst_dim)
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
    # model.hurst.fparam[0, :] = fparam[:]
    model.hurst.ChangeParameters(fparam, finter)
    model.NormalizeModel()

    Tau0[expe, :] = topo.fparam[0, :]
    Beta0[expe, :] = hurst.fparam[0, :]

    # Computation of the theoretical semi-variogram
    model.ComputeApproximateSemiVariogram(lags)
    w0 = np.zeros(model.svario.values.size)
    w0[:] = model.svario.values[:, 0]

    if noise == 1:
        s0 = np.random.rand() * np.min(w0)  # Noise variance
    else:
        s0 = 0

    # Semivariogram to be fitted (theoretical / empirical, noise / no noise)
    if Tvario:
        w = np.zeros(w0.shape)
        w[:] = w0[:]
        if noise == 1:
            w = w + s0
    else:
        # Simulate a field realization
        z = model.Simulate(coord)
        if noise == 1:
            z.values = z.values +\
                np.sqrt(s0) * rng.standard_normal(z.values.shape)

        # Compute the empirical semi-variogram
        evario = z.ComputeEmpiricalSemiVariogram(lags)
        w = evario.values[:, 0]
        # Evaluate the estimation error
        err_est += np.mean(np.abs(w0 + s0 - w))

    # Initial variogram fitting with varproj
    t0 = time.perf_counter()
    emodel0, wt = FitVariogram(model0, lags, w, myparam)
    t1 = time.perf_counter()
    time_c1 += t1 - t0

    emodel0.topo.Evaluate(fintermid)
    emodel0.hurst.Evaluate(fintermid)
    emodel1.topo.fparam[0, :] = emodel0.topo.values[0, :]
    emodel1.hurst.fparam[0, :] = emodel0.hurst.values[0, :]
    Tau1[expe, :] = emodel1.topo.fparam[0, :]
    Beta1[expe, :] = emodel1.hurst.fparam[0, :]

    # Variogram fitting with varprox
    myparam.threshold_reg = 4
    t0 = time.perf_counter()
    emodel2, w1 = FitVariogram(model0, lags, w, myparam)
    t1 = time.perf_counter()
    time_c2 += t1 - t0

    emodel2.topo.Evaluate(fintermid)
    emodel2.hurst.Evaluate(fintermid)
    emodel3.topo.fparam[0, :] = emodel2.topo.values[0, :]
    emodel3.hurst.fparam[0, :] = emodel2.hurst.values[0, :]
    Tau2[expe, :] = emodel3.topo.fparam[0, :]
    Beta2[expe, :] = emodel3.hurst.fparam[0, :]

    print('Running experiments = {:3d} / {:3d}.'.format(expe + 1, Nbexpe))

    if save:
        data_filename = "results_" + str(expe+1) + ".pickle"
        with open(data_filename, "wb") as f:
            pickle.dump([Beta1, Tau1, Beta2, Tau2, model0, emodel0, emodel1,
                         emodel2, emodel3], f)


print('\nExperiment report:')
print(' - Number of coefficients (beta={:d}, tau={:d})'.format(hurst_dim,
                                                               hurst_dim))
print(' - Radial precision: {:e}'.format(np.pi / hurst_dim))
print(' - Theoretical variogram: {}'.format(Tvario))
print(" - Reg param (beta) = {:.3E}".format(myparam.reg_param))
print(" - Reg param (tau) =  {:.3E}".format(myparam.alpha))
print_report("1) Varproj", Beta1, Tau1, Beta0, Tau0, time_c1, Nbexpe)
print_report("2) Varprox", Beta2, Tau2, Beta0, Tau0, time_c2, Nbexpe)

