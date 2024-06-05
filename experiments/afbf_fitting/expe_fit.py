# -*- coding: utf-8 -*-
r"""Fitting variogram of an anisotropic fractional Brownian field:
    Part 2. Variogram fitting.
"""
import time
import numpy as np
from afbf import coordinates, perfunction, tbfield
from varprox.models.model_afbf import FitVariogram
from afbf.Simulation.TurningBands import LoadTBField
from afbf.Classes.SpatialData import LoadSdata
from numpy.random import default_rng
from varprox import Parameters
from param_expe_8_evario import params
from os import path


# Data repertory.
home_dir = "/home/frichard/Recherche/Python/varprox/"
# home_dir = "C:/Users/frede/Nextcloud/Synchro/Recherche/Python/varprox/"


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
param_opti.verbose = True

# Lags where to compute the semi-variogram
lags = coordinates()
lags.DefineSparseSemiBall(param.grid_dim)
lags.N = param.grid_dim * 2

time_c1 = 0
time_c2 = 0
for _ in range(2):
    for expe in range(param.Nbexpe):
        caseid = str(expe + 100)
        caseid = caseid[1:]
        file_in = home_dir + param.data_in + caseid
        file_out = home_dir + param.data_out + caseid

        if path.exists(file_out + "-varproj-hurst.pickle") is False:
            optim = "varproj"
        elif path.exists(file_out + "-varprox-hurst.pickle") is False:
            optim = "varprox"
        else:
            optim = None

        if optim is not None:
            print('Running experiments = {:3d} / {:3d}.'.format(expe,
                                                                param.Nbexpe - 1))

            # Semivariogram to be fitted (theoretical / empirical, noise / no noise)
            if param.Tvario:
                # Load the groundtruth model.
                model = LoadTBField(file_in)
                # Compute the theoretical semi-variogram
                model.ComputeApproximateSemiVariogram(lags)
                w = np.zeros(model.svario.values.size)
                w[:] = model.svario.values[:, 0]

                if param.noise == 1:
                    s0 = np.random.rand() * np.min(w)
                    w = w + s0
                else:
                    s0 = 0
            else:
                # Load a field realization
                z = LoadSdata(file_in + "-imag")
                if param.crop is not None:
                    z.values.reshape(z.M)[0:param.crop, 0:param.crop]
                    z.M = np.array([param.crop, param.crop])
                    z.values = np.reshape(z.values, (np.prod(z.M), 1))
                if param.noise == 1:
                    # Load the groundtruth model.
                    model = LoadTBField(file_in)
                    # Compute the theoretical semi-variogram
                    model.ComputeApproximateSemiVariogram(lags)
                    w = np.zeros(model.svario.values.size)
                    w[:] = model.svario.values[:, 0]
                    s0 = np.random.rand() * np.min(w)  # Noise variance.
                    z.values = z.values +\
                        np.sqrt(s0) * rng.standard_normal(z.values.shape)
                else:
                    s0 = 0

                # Compute the empirical semi-variogram
                evario = z.ComputeEmpiricalSemiVariogram(lags)
                w = evario.values[:, 0]

            # Initialize the estimation model.
            topo0 = perfunction('step', param.topo_dim)
            hurst0 = perfunction('step', param.hurst_dim)
            model0 = tbfield('Estimation model', topo0, hurst0)
            param_opti.alpha = 1e-1 * np.mean(np.power(w, 2))
            if optim == "varproj":
                # Variogram fitting with varpro.
                param_opti.reg.name = None
            elif optim == "varprox":
                # model0 = LoadTBField(file_out + "-varproj")
                # Variogram fitting with varprox.
                param_opti.threshold_reg = param.hurst_dim
                param_opti.reg.name = "tv-1d"
                param_opti.reg.weight = 1e-1 * np.mean(np.power(w, 2))

            t0 = time.perf_counter()
            emodel, wt = FitVariogram(model0, lags, w, param_opti)
            t1 = time.perf_counter()

            emodel.Save(file_out + "-" + optim)
