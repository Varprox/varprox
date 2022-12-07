# -*- coding: utf-8 -*-
r"""
Experiments of fitting variogram of an anisotropic fractional Brownian field.
OBSOLETE. use expe_afbf_2

"""
import time
import numpy as np
from afbf import coordinates, perfunction, tbfield
from varprox.models.model_afbf import BasisFunctions, SemiVariogram,\
    MultiGridFitting

Nbexpe = 5  # Number of experiments.
N = 30  # size of the grid for the definition of the semi-variogram.
M = 512  # size of field realization.
step = 1  # grid step.
K = 3  # For defining the number of parameters
Tvario = True  # If true, use the theoretical semi-variogram.
ftype = 'step'

# Reference model.
K = 2**(K - 1)  # Number of parameters.
J = K

topo0 = perfunction(ftype, J)
hurst0 = perfunction(ftype, K)
Jinter = topo0.finter.size
Kinter = hurst0.finter.size
It = np.linspace(- np.pi / 2, np.pi / 2, Jinter + 1, True)[1:]
Ih = np.linspace(- np.pi / 2, np.pi / 2, Kinter + 1, True)[1:]

Z0 = tbfield('reference', topo0, hurst0)
topo0.ChangeParameters(topo0.fparam, It)
hurst0.ChangeParameters(hurst0.fparam, Ih)
tau0 = topo0.fparam.reshape((J,))
beta0 = hurst0.fparam.reshape((K,))


# Turning-band angles.
phi = Z0.tb.Kangle
dphi = np.diff(phi)
phi = phi[1:]
phi = np.expand_dims(phi, axis=0)
dphi = np.expand_dims(dphi, axis=1)


# Grid for the computation of the semi-variogram.
lags = coordinates()
lags.DefineUniformGrid(N, step, True)
xy = lags.xy[np.random.permutation(lags.xy.shape[0]), :]
sc = np.power(xy[:, 0], 2) + np.power(xy[:, 1], 2)
sc, ind = np.unique(sc, return_index=True)
xy = xy[ind, :]
lags.DefineNonUniformLocations(xy)
lags.N = N

# Ffun and Dfun parameters.
f = np.power(xy @ np.concatenate((np.cos(phi), np.sin(phi)), axis=0), 2) / N**2
ind = np.nonzero(f > 0)
lf = np.zeros(f.shape)
lf[ind] = np.log(f[ind])

# Basis function for the computation of the theoretical semi-variogram.
T = BasisFunctions(topo0, phi) * dphi
B = BasisFunctions(hurst0, phi)

Tau0 = np.zeros((Nbexpe, tau0.size))
Beta0 = np.zeros((Nbexpe, beta0.size))
Tau1 = np.zeros((Nbexpe, tau0.size))
Beta1 = np.zeros((Nbexpe, beta0.size))
err_a = 0
err_r = 0
varerr = 0
time_c = 0
for expe in range(Nbexpe):
    # Define a new model.
    np.random.seed(expe)
    topo0.ChangeParameters(None, It)
    hurst0.ChangeParameters(None, Ih)
    Tau0[expe, :] = topo0.fparam[0, :]
    Beta0[expe, :] = hurst0.fparam[0, :]

    # Computation of the theoretical semi-variogram.
    w0 = SemiVariogram(tau0, beta0, f, lf, T, B)

    # Theoretical vs empirical semivariogram.
    if Tvario:
        w = w0
    else:
        # Empirical semi-variogram.
        coord = coordinates(M)
        coord.N = N
        z = Z0.Simulate(coord)
        evario = z.ComputeEmpiricalSemiVariogram(lags)
        w1 = evario.values
        varerr += np.mean(np.abs(w0 - w1))
        w = w1

    # Variogram fitting.
    t0 = time.time()

    tau1, beta1 = MultiGridFitting(xy, phi, dphi, w, N,
                                   ftype,
                                   maxnpar=K, multigrid=False,
                                   maxit=10000, gtol=0.001,
                                   verbose=0)
    t1 = time.time()
    time0 = t1 - t0
    time_c += time0
    print('expe %d / %d: %e (sec)' % (expe + 1, Nbexpe, time0))
    Tau1[expe, :] = tau1[:]
    Beta1[expe, :] = beta1[:]

    # Computation of absolute and relative errors on variograms.
    we = SemiVariogram(tau1, beta1, f, lf, T, B)
    err = np.mean(np.power(we - w0, 2))
    err_a += err
    err_r += err / np.mean(np.power(w0, 2)) * 100


print('Experiment report:')
print('Number of coefficients (beta=%d, tau=%d)' % (J, K))
print('Theoretical variogram: %d' % (Tvario))
if Tvario is False:
    print('Variogram estimation error: %e' % (varerr / Nbexpe))
print('Mean execution time: %e (sec)' % (time_c / Nbexpe))
print('Error on variogram:')
print('Absolute / relative errors: %e, %e' % (err_a, err_r))
print('Error on coefficients:')
print('beta=%e, tau=%e' % (np.mean(np.absolute(Beta1 - Beta0), axis=None),
                           np.mean(np.absolute(Tau1 - Tau0), axis=None)))


# For test purpose.

# tau = np.zeros(tau0.shape)
# tau[:] = tau0[:]
# beta = np.zeros(beta0.shape)
# beta[:] = beta0[:]

# bounds = (0, np.inf)
# tau = np.random.rand(tau.size)
# pb = minimize(beta0, tau, w0, Ffun, DFfun, bounds, bounds, f, lf, T, B)
# tau1 = pb.argmin_h_y(beta0)
# print('argmin tau')
# print('Error:')
# print(np.mean(np.absolute(tau1 - tau0)))
# # print(tau0)
# # print(tau1)

# beta = np.random.rand(beta.size)
# pb = minimize(beta, tau0, w0, Ffun, DFfun, bounds, bounds, f, lf, T, B)
# beta1 = pb.argmin_h_x(beta, gtol=1e-6, maxit=10000)
# print('argmin beta')
# print('Error:')
# print(np.mean(np.absolute(beta1 - beta0)))
# # print(beta0)
# # print(beta1)
