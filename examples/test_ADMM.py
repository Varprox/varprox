# -*- coding: utf-8 -*-
r"""
======================================
Anisotropic fractional Brownian field.
======================================

"""
import numpy as np
# from varprox import minimize
from afbf import coordinates, perfunction, tbfield
from varprox.models.model_afbf import FitVariogramADMM
import matplotlib.pyplot as plt


N = 30  # size of the grid for the definition of the semi-variogram.
step = 1  # grid step.
M = 256  # size of field realization.
K = 2**2  # Number of parameters for the Hurst function.
J = K  # Number of parameters for the topothesy function
ftype = 'step'
scalemin = 0

# Definition of the reference model.
topo = perfunction(ftype, J)
hurst = perfunction(ftype, K)
Jinter = topo.finter.size
Kinter = hurst.finter.size
It = np.linspace(- np.pi / 2, np.pi / 2, Jinter + 1, True)[1:]
Ih = np.linspace(- np.pi / 2, np.pi / 2, Kinter + 1, True)[1:]
topo.ChangeParameters(topo.fparam, It)
hurst.ChangeParameters(hurst.fparam, Ih)
model = tbfield('reference', topo, hurst)

# Grid for the computation of the semi-variogram.

# Lags for the computation of the semi-variogram.
lags = coordinates()
lags.DefineSparseSemiBall(N, step)
sc = np.sqrt(np.power(lags.xy[:, 0], 2) + np.power(lags.xy[:, 1], 2))
ind = np.nonzero(sc > scalemin)
lags.xy = lags.xy[ind[0], :]
sc = sc[ind[0]]
lags.N = N

J = topo.fparam.size
K = hurst.fparam.size


tau0 = topo.fparam.reshape((J,))
beta0 = hurst.fparam.reshape((K,))

# Computation of the theoretical semi-variogram.
model.ComputeApproximateSemiVariogram(lags)
w0 = np.zeros(model.svario.values.shape)
w0[:] = model.svario.values[:]


fittedModel, fittedVario = FitVariogramADMM(model, lags, w0, multigrid=True, 
                                  maxit=500,gtol=1e-5,verbose=1,alpha=0.1)


# print(tau1)
# print(tau0)
# print(beta1)
# print(beta0)
print('Absolutre l1 Error:')
print(np.mean(np.absolute(fittedModel.hurst.fparam - beta0)))
print(np.mean(np.absolute(fittedModel.topo.fparam- tau0)))

print('Relative mean Error l2:')
print(np.sqrt(np.sum( (fittedModel.hurst.fparam - beta0)**2) / np.sum(beta0**2) ))
print(np.sqrt(np.sum((fittedModel.topo.fparam- tau0)**2) / np.sum(tau0**2) ))


# import matplotlib.pyplot as plt
# var0 = SemiVariogram(tau0, beta0, f, lf, T, B)
# var1 = SemiVariogram(tau1, beta1, f, lf, T, B)

model.ComputeApproximateSemiVariogram(lags)
model.svario.Display(1)

fittedModel.ComputeApproximateSemiVariogram(lags)
fittedModel.svario.Display(2)


