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
    Part 1. Data simulation.
"""
import numpy as np
from afbf import coordinates, perfunction, tbfield, process
from numpy.random import default_rng, seed
from param_expe import params
from os import path
from varprox import Parameters


# Repetory for data
home_dir = "./"

# Initialization a new random generator
rng = default_rng()

# Optimisation parameters.
param = params()
param_opti = Parameters()
param_opti.load("param_optim.ini")
param_opti.noise = 0
param_opti.threshold_reg = np.Inf
param_opti.verbose = False
param_opti.maxit = 100

# Definition of the reference model
topo = perfunction('step', param.topo_dim)
hurst = perfunction('step', param.hurst_dim)
finter = np.linspace(- np.pi / 2, np.pi / 2, hurst.finter.size + 1, True)
fintermid = (finter[0:-1] + finter[1:]) / 2
finter = finter[1:]
model = tbfield('reference', topo, hurst)
tb = model.tb

# Definition of a fbm to sample the Hurst function
fbm = process()
fbm.param = 0.9

lags = coordinates()
lags.DefineSparseSemiBall(param.grid_dim)
lags.N = param.grid_dim * 2

coord = coordinates(param.N)
coord.N = param.grid_dim * 2
for expe in range(param.Nbexpe):
    caseid = str(expe + 100)
    caseid = caseid[1:]
    filename = home_dir + param.data_in + caseid

    if path.exists(filename + "-imag.pickle") is False:
        print('Running experiments = {:3d} / {:3d}.'.format(expe,
                                                            param.Nbexpe - 1))

        seed(expe)
        # Change model parameters.
        fbm.Simulate(param.hurst_dim)
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

        # Update the model.
        model.hurst.ChangeParameters(fparam, finter)
        model.NormalizeModel()
        model.topo.fparam = model.topo.fparam * param.enhan_factor
        model.DisplayParameters()

        # Simulate a field realization.
        z = model.Simulate(coord)
        z.Display()

        # Save model and image.
        z.Save(filename + "-imag")
        model.Save(filename)