#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
# Credits
# #######
#
# Copyright(c) 2025-2025
# ----------------------
#
# * Institut de Mathématiques de Marseille <https://www.i2m.univ-amu.fr/>
# * Université d'Aix-Marseille <http://www.univ-amu.fr/>
# * Centre National de la Recherche Scientifique <http://www.cnrs.fr/>
#
# Contributors
# ------------
#
# * `Arthur Marmin <mailto:arthur.marmin@univ-amu.fr>`_
# * `Frédéric Richard <mailto:frederic.richard@univ-amu.fr>`_
#
#
# * This module is part of the package Varprox.
#
# Licence
# -------
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ######### COPYRIGHT #########
r"""Fitting variogram of an anisotropic fractional Brownian field:
    Part 2. Variogram fitting.
"""
import time
import numpy as np
from afbf import coordinates, perfunction, tbfield
from afbfest.model_afbf import FitVariogram
from afbf.Simulation.TurningBands import LoadTBField
from afbf.Classes.SpatialData import LoadSdata
from numpy.random import default_rng
from varprox import Parameters
from param_expe import params
from os import path


# Data repertory.
home_dir = "./"


# Experience parameters.
param = params()
reg_weight = str(param.reg_param)

# Initialization a new random generator
rng = default_rng()

# Lags where to compute the semi-variogram
lags = coordinates()
lags.DefineSparseSemiBall(param.grid_dim)
lags.N = param.grid_dim * 2

time_c1 = 0
time_c2 = 0
for _ in range(3):
    for expe in range(7, param.Nbexpe):
        caseid = str(expe + 100)
        caseid = caseid[1:]
        file_in = home_dir + param.data_in + caseid
        file_out = home_dir + param.data_out + caseid

        if path.exists(file_out + "-vanilla-hurst.pickle") is False:
            optim = "varproj"
            resname = "vanilla"
        elif path.exists(file_out + "-varproj-hurst.pickle") is False:
            optim = "varproj"
            resname = "varproj"
        elif path.exists(file_out + "-varprox-hurst.pickle") is False:
            optim = "varprox"
            resname = "varprox"
        else:
            optim = None

        if optim is not None:
            head = 'Running experiments = {:3d} / {:3d}.'
            print(head.format(expe, param.Nbexpe - 1))

            # Setting parameters for model and optimisation.
            param_opti = Parameters()
            param_opti.load("param_optim.ini")
            param_opti.multigrid = param.multigrid
            param_opti.noise = param.noise
            param_opti.alpha = param.alpha
            param_opti.reg.order = param.order

            # Semivariogram to be fitted
            # (theoretical / empirical, noise / no noise)
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
                z.values = z.values.reshape(z.M)[0:param.crop, 0:param.crop]
                z.M = np.array([param.crop, param.crop])
                z.values = np.reshape(z.values, (np.prod(z.M), 1))
                if param.noise == 1:
                    # Load the groundtruth model.
                    model = LoadTBField(file_in)
                    # Compute the theoretical semi-variogram
                    model.ComputeApproximateSemiVariogram(lags)
                    w = np.zeros(model.svario.values.size)
                    w[:] = model.svario.values[:, 0]
                    w = w / np.max(w)
                    s0 = rng.uniform() * np.min(w)  # Noise variance.
                    z.values = z.values +\
                        np.sqrt(s0) * rng.standard_normal(z.values.shape)
                else:
                    s0 = 0

                # Compute the empirical semi-variogram
                # evario = z.ComputeQuadraticVariations(lags, order=1)
                evario = z.ComputeEmpiricalSemiVariogram(lags)
                w = evario.values[:, 0]

            # Initialize the estimation model.
            topo0 = perfunction('step', param.topo_dim)
            hurst0 = perfunction('step', param.hurst_dim)
            model0 = tbfield('Estimation model', topo0, hurst0)
            if optim == "varproj":
                # Variogram fitting with varpro.
                param_opti.reg.name = None
                if resname == "vanilla":
                    param.alpha = 0
            elif optim == "varprox":
                # Variogram fitting with varprox.
                param_opti.reg.name = "tv-1d"
                param_opti.threshold_reg = param.threshold_reg
                param_opti.reg.weight = param.reg_param

            t0 = time.perf_counter()
            emodel, wt = FitVariogram(model0, lags, w, param_opti)
            t1 = time.perf_counter()

            emodel.Save(file_out + "-" + resname)
