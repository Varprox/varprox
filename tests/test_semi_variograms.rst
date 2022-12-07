Test the computation of the semi-variograms for afbf increments.
================================================================

>>> import numpy as np
>>> from afbf import coordinates, perfunction, tbfield
>>> from matplotlib import pyplot as plt
>>> from varprox.models.model_afbf import FitVariogram
>>> # Parameters to set the lags where to compute the semi-variogram.
>>> M = 512  # Image size.
>>> N = 30  # size of the grid for the definition of the semi-variogram.
>>> step = 2  # step for grid definition
>>> # Model parameters.
>>> J = K = 2  # Number of parameters for the Hurst function.
>>> ftype = "step"  # Type of representation for Hurst and topothesy functions.
>>> # Reference model.
>>> np.random.seed(1)
>>> topo0 = perfunction('step', J)
>>> hurst0 = perfunction('step', K)
>>> Jinter = topo0.finter.size
>>> Kinter = hurst0.finter.size
>>> It = np.linspace(- np.pi / 2, np.pi / 2, Jinter + 1, True)[1:]
>>> topo0.ChangeParameters(None, It)
>>> hurst0.ChangeParameters(None, It)
>>> model0 = tbfield('reference', topo0, hurst0)
>>> # Coordinates where to compute the semi-variograms.
>>> lags = coordinates()
>>> lags.DefineUniformGrid(N, step, True)
>>> xy = np.zeros(lags.xy.shape)
>>> xy[:] = lags.xy[:]
>>> # Coordinates where to simulate images.
>>> coord = coordinates()
>>> coord.DefineUniformGrid(M)
>>> # Increment lag.
>>> kj = 70
>>> k = np.zeros((2,))
>>> k[:] = lags.xy[kj, :]
>>> scalemax = np.sqrt(np.sum(np.power(k, 2)))
>>> lags.N = M


Compute the semi-variograms with pyafbf tools.
----------------------------------------------

>>> model0.ComputeApproximateSemiVariogram(lags)
>>> v = model0.svario.values[:, 0]
>>> vk = model0.svario.values[kj, 0]
>>> w = 2 * (v + vk)
>>> lags.xy[:, 0] = xy[:, 0] + k[0]
>>> lags.xy[:, 1] = xy[:, 1] + k[1]
>>> model0.ComputeApproximateSemiVariogram(lags)
>>> w = w - model0.svario.values[:, 0]
>>> lags.xy[:, 0] = xy[:, 0] - k[0]
>>> lags.xy[:, 1] = xy[:, 1] - k[1]
>>> model0.ComputeApproximateSemiVariogram(lags)
>>> w = w - model0.svario.values[:, 0]
>>> lags.xy[:, 0] = xy[:, 0]
>>> lags.xy[:, 1] = xy[:, 1]

Test the semi-variograms computed with FitVariogram.
----------------------------------------------------

>>> model, v2 = FitVariogram(model0, lags, v, 0, None, False, 0, 0.1, 0)
>>> model, w2 = FitVariogram(model0, lags, w, 0, k, False, 0, 0.1, 0)
>>> np.mean(np.abs(v2 - v)) < 10e-16
True
>>> np.mean(np.abs(w2 - w)) < 10e-16
True

Test their adequacy with the empirical semi-variograms.
-------------------------------------------------------

>>> simu = model.Simulate(coord)
>>> simu_inc = simu.ComputeIncrements(k[0], k[1])
>>> ve = simu.ComputeEmpiricalSemiVariogram(lags).values[:, 0]
>>> we = simu_inc.ComputeEmpiricalSemiVariogram(lags).values[:, 0]
>>> np.mean(np.abs(ve - v)) < 10e-2 
True
>>> np.mean(np.abs(we - w)) < 10e-2
True

