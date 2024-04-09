# -*- coding: utf-8 -*-
r"""
Experiments of fitting variogram of an anisotropic fractional Brownian field.
"""
import time
import numpy as np
from afbf import coordinates, perfunction, tbfield
from varprox.models.model_afbf import FitVariogram
from afbf.Simulation.TurningBands import LoadTBField
from afbf.Classes.SpatialData import LoadSdata
from numpy.random import default_rng
from varprox import Parameters
from param_expe import params

# Data repertory.
home_dir = "/home/frichard/Recherche/Python/varprox/"
data_in = "data/afbf_fitting/"
data_out = "experiments/afbf_fitting/results/"


# Experience parameters.
param = params()

# Initialization a new random generator
rng = default_rng()

# Setting parameters for model and optimisation.
param_opti = Parameters()
param_opti.load("param_optim.ini")
param_opti.multigrid = param.multigrid
param_opti.noise = param.noise
param_opti.threshold_reg = np.Inf
param_opti.reg.name = None


# Lags where to compute the semi-variogram
lags = coordinates()
lags.DefineSparseSemiBall(param.grid_dim)
lags.N = 80  # param.N
sc = np.sqrt(np.power(lags.xy[:, 0], 2) + np.power(lags.xy[:, 1], 2))

err_fit = err_est = err_tes = 0
time_c1 = 0
time_c2 = 0
for expe in range(1):  # Nbexpe):
    caseid = str(expe + 100)
    caseid = caseid[1:]
    file_in = home_dir + data_in + caseid
    file_out = home_dir + data_out + caseid

    # Load the groundtruth model.
    model = LoadTBField(file_in)

    # Computation of the theoretical semi-variogram
    model.ComputeApproximateSemiVariogram(lags)
    w0 = np.zeros(model.svario.values.size)
    w0[:] = model.svario.values[:, 0]

    if param.noise == 1:
        s0 = np.random.rand() * np.min(w0)  # Noise variance
    else:
        s0 = 0

    # Semivariogram to be fitted (theoretical / empirical, noise / no noise)
    if param.Tvario:
        w = np.zeros(w0.shape)
        w[:] = w0[:]
        if param.noise == 1:
            w = w + s0
    else:
        # Load a field realization
        z = LoadSdata(file_in + "-imag")
        if param.noise == 1:
            z.values = z.values +\
                np.sqrt(s0) * rng.standard_normal(z.values.shape)

        # Compute the empirical semi-variogram
        evario = z.ComputeEmpiricalSemiVariogram(lags)
        w = evario.values[:, 0]
        # Evaluate the estimation error
        err_est += np.mean(np.abs(w0 + s0 - w))

    # Estimation model.
    topo0 = perfunction('step', 128)
    hurst0 = perfunction('step', 128)
    model0 = tbfield('Estimation model', topo0, hurst0)

    # Initial variogram fitting with varpro.
    t0 = time.perf_counter()
    emodel_varproj, wt = FitVariogram(model0, lags, w, param_opti)
    t1 = time.perf_counter()
    time_c1 += t1 - t0

    # Variogram fitting with varprox.
    # param_opti.threshold_reg = 4
    # param_opti.reg.name = "tv-1d"
    # param_opti.reg.weight = 0.01
    # t0 = time.perf_counter()
    # emodel_varprox, w1 = FitVariogram(model0, lags, w, param_opti)
    # t1 = time.perf_counter()
    # time_c2 += t1 - t0

    print('Running experiments = {:3d} / {:3d}.'.format(expe + 1,
                                                        param.Nbexpe))

    # if save:
    #     emodel_varproj.Save(file_out + "-varproj")
    #     emodel_varprox.Save(file_out + "-varprox")
