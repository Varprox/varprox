# -*- coding: utf-8 -*-
r"""Fitting variogram of an anisotropic fractional Brownian field:
    Part 1. Data simulation.
"""
import numpy as np
from afbf import coordinates, perfunction, tbfield, process
from numpy.random import default_rng
from param_expe import params


# Repetory for data
home_dir = "/home/frederic/Recherche/Python/varprox/"
# home_dir = "C:/Users/frede/Nextcloud/Synchro/Recherche/Python/varprox/"
data_out = "data/afbf_fitting/"

# Initialization a new random generator
rng = default_rng()

param = params()

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


coord = coordinates(param.N)
coord.N = param.grid_dim * 2

for expe in range(param.Nbexpe):
    caseid = str(expe + 100)
    caseid = caseid[1:]
    filename = home_dir + data_out + caseid

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
