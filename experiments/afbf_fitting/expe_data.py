# -*- coding: utf-8 -*-
r"""Build a database for experiments on variogram fitting.
"""
import numpy as np
from afbf import coordinates, perfunction, tbfield, process
from numpy.random import default_rng

# Experiment parameters
# Number of experiments
Nbexpe = 2
# Number of parameters for the Hurst and topothesy functions
hurst_dim = topo_dim = 512
# Image size.
N = 512
# Repetory for data
home_dir = "/home/frichard/Recherche/Python/"
data_out = "varprox/data/afbf_fitting/"

# Initialization a new random generator
rng = default_rng()

# Definition of the reference model
topo = perfunction('step', topo_dim)
hurst = perfunction('step', hurst_dim)
finter = np.linspace(- np.pi / 2, np.pi / 2, hurst.finter.size + 1, True)
fintermid = (finter[0:-1] + finter[1:]) / 2
finter = finter[1:]
model = tbfield('reference', topo, hurst)
tb = model.tb

# Definition of a fbm to sample the Hurst function
fbm = process()
fbm.param = 0.9


coord = coordinates(N)
coord.N = N

for expe in range(Nbexpe):
    caseid = str(expe + 100)
    caseid = caseid[1:]
    filename = home_dir + data_out + caseid

    # Change model parameters.
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

    # Update the model.
    model.hurst.ChangeParameters(fparam, finter)
    model.NormalizeModel()
    model.topo.fparam = model.topo.fparam * 10e4
    model.DisplayParameters()

    # Simulate a field realization.
    z = model.Simulate(coord)
    z.Display()

    # Save model and image.
    z.Save(filename + "-imag")
    model.Save(filename)
