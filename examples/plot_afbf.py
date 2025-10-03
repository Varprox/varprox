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
r"""
====================================================================
Estimation of parameters of an anisotropic fractional Brownian field
====================================================================

Anisotropic fractional Brownian fields are random fields whose properties
are characterized by two functional parameters, namely the Hurst function
and the topothesy functions. These two functions determined the anisotropy
and regularity of the field;
see `PyAFBF <https://github.com/fjprichard/PyAFBF>`__.
for more explanations. In this example, we use varprox
to estimate the two functional parameters of the field.

. note::
    This example requires the installation of the
    `PyAFBFest <https://github.com/fjprichard/PyAFBFest>`__.
"""
import numpy as np
from afbf import coordinates, perfunction, tbfield, process
from numpy.random import default_rng, seed
from varprox import Parameters
from afbfest.model_afbf import FitVariogram


def Field_Definition(param):
    """Definition of the reference model.
    """
    topo = perfunction('step', param.topo_dim)
    hurst = perfunction('step', param.hurst_dim)
    finter = np.linspace(- np.pi / 2, np.pi / 2, hurst.finter.size + 1, True)
    finter = finter[1:]
    model = tbfield('Reference model', topo, hurst)

    # Define model parameters.
    fbm = process()
    fbm.param = 0.9
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
    model.topo.fparam = model.topo.fparam * 10
    model.hurst.fname = "Original Hurst function"
    model.topo.fname = "Original topothesy"

    return model


# Initialization a new random generator
rng = default_rng()
seed(88)

# Set some parameters.
param = Parameters()
param.load("plot_afbf.ini")
param.noise = 0
param.hurst_dim = 64  # Dimension of the function parametrization.
param.topo_dim = 64
param.N = 512  # Image size.
param.grid_dim = 20  # Size of the grid to compute quadratic variations.
param.multigrid = True  # To use a multigrid optimization approach.
param.threshold_reg = 32  # Grid scale at which the penalization is used.

#: Define the field model.
model = Field_Definition(param)

#: Simulate a field realization.
lags = coordinates()
lags.DefineSparseSemiBall(param.grid_dim)
lags.N = param.grid_dim * 2
coord = coordinates(param.N)
coord.N = param.grid_dim * 2
z = model.Simulate(coord)
z.Display()
model.DisplayParameters()

#: Compute the empirical semi-variogram.
evario = z.ComputeEmpiricalSemiVariogram(lags)
w = evario.values[:, 0]

#: Estimate model parameters.
topo0 = perfunction('step', param.topo_dim)

hurst0 = perfunction('step', param.hurst_dim)

model0 = tbfield('Estimation model', topo0, hurst0)
emodel, wt = FitVariogram(model0, lags, w, param)
emodel.name = "Estimated model"
emodel.hurst.fname = 'Estimated Hurst function'
emodel.topo.fname = 'Estimated topothesy'
emodel.DisplayParameters()
