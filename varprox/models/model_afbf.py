# -*- coding: utf-8 -*-
r"""
Tools for computing the semi-variogram of an anisotropic fractional
Brownian field and applying the fitting method.
"""
import numpy as np
from scipy.linalg import circulant
from afbf import perfunction, tbfield
from varprox import Minimize


def BasisFunctions(fun, t):
    """Values of the basis functions of the periodic function perfun at
       positions phi.

       :param basis: an array where values of the mth basis function
           of the representation are on the mth column.
       :type basis: :ref:`ndarray`
    """
    fun.Evaluate(t)
    return fun.basis.T


def SemiVariogram(tau, beta, f, lf, T, B, noise=1):
    """Compute the semi-variogram of an AFBF or its increment field.
    """
    return Ffun(beta, f, lf, T, B, noise) @ tau


def Ffun(beta, f, lf, T, B, noise=1):
    return np.concatenate((np.ones((f.shape[0], noise)),
                           0.5 * np.power(f, B @ beta) @ T), axis=1)


def DFfun(beta, tau, f, lf, T, B, noise=1):
    DF = np.zeros((f.shape[0], T.shape[1] + noise, B.shape[1]))

    v = 0.5 * lf * np.power(f, B @ beta)
    for j in range(noise, DF.shape[1]):
        for k in range(DF.shape[2]):
            DF[0:f.shape[0], j, k] = v @ (T[:, j - noise] * B[:, k])

    return np.swapaxes(DF, 1, 2) @ tau


def FitVariogram(model, lags, w, param):
    """Fit the field variogram using a coarse-to-fine multigrid strategy.
    """
    # Regularization parameters.
    reg_name = param.reg.name

    if model.hurst.ftype != "step" or model.topo.ftype != "step":
        raise ValueError("FitVariogram: only runs for step functions.")

    # Ffun and Dfun parameters
    # Turning-band angles
    phi = np.zeros(model.tb.Kangle.shape)
    phi[:] = model.tb.Kangle[:]
    dphi = np.diff(phi)
    phi = phi[1:]
    phi = np.expand_dims(phi, axis=0)
    dphi = np.expand_dims(dphi, axis=1)
    # Coordinates where variograms are computed
    xy = lags.xy
    N = lags.N

    csphi = np.concatenate((np.cos(phi), np.sin(phi)), axis=0)
    f = np.power(xy @ csphi, 2) / N**2
    lf = np.zeros(f.shape)
    ind = np.nonzero(f > 0)
    lf[ind] = np.log(f[ind])

    # Number of model parameters
    npar0_tau = model.topo.fparam.size
    npar0_beta = model.hurst.fparam.size

    if param.multigrid:
        hurst = perfunction(model.hurst.ftype, param=1)
        topo = perfunction(model.topo.ftype, param=1)
    else:
        hurst = perfunction(model.hurst.ftype, param=npar0_beta)
        topo = perfunction(model.topo.ftype, param=npar0_tau)
        hurst.fparam[:] = model.hurst.fparam[:]
        hurst.finter[:] = model.hurst.finter[:]
        topo.fparam[:] = model.topo.fparam[:]
        topo.finter[:] = model.topo.finter[:]

    beta = np.zeros((hurst.fparam.size,))
    beta[:] = hurst.fparam[0, :]

    stop = False
    while stop is False:
        B = BasisFunctions(hurst, phi)
        T = BasisFunctions(topo, phi) * dphi

        # Cancel the tv regularization if only one parameter is involved.
        if reg_name == "tv-1d":
            if beta.size < param.threshold_reg:
                param.reg.name = None
            else:
                param.reg.name = reg_name

        pb = Minimize(beta, w, Ffun, DFfun, f, lf, T, B, param.noise)
        pb.params = param

        if param.verbose:
            print("Nb param: Hurst={:d}, Topo={:d}".format(
                hurst.fparam.size, topo.fparam.size))
            print("Tol = {:.5e}, Nepochs = {:d}".format(param.gtol_h,
                                                        param.maxit))

        beta, tau = pb.argmin_h()
        topo.fparam[0, :] = tau[param.noise:]
        hurst.fparam[0, :] = beta[:]

        stop = True
        if param.multigrid:
            npar = (tau.size - param.noise) * 2
            if npar <= npar0_tau:
                stop = False
                topo0 = topo
                topo = perfunction(model.topo.ftype, param=npar)
                Iv = np.linspace(-np.pi / 2, np.pi / 2, npar + 1, True)[1:]
                topo0.Evaluate(Iv)
                topo.fparam[0, :] = topo0.values[0, :]

            npar = beta.size * 2
            if npar <= npar0_beta:
                stop = False
                npar_beta = npar
                hurst0 = hurst
                hurst = perfunction(model.hurst.ftype, param=npar_beta)
                Iv = np.linspace(-np.pi / 2, np.pi / 2, npar_beta + 1, True)[1:]
                hurst0.Evaluate(Iv)
                hurst.fparam[0, :] = hurst0.values[0, :]
                beta = np.zeros((hurst.fparam.size,))
                beta[:] = hurst.fparam[0, :]

    emodel = tbfield("Estimated model", topo, hurst, model.tb)
    if param.noise == 1:
        emodel.noise = tau[0]
    else:
        emodel.noise = 0

    return (emodel, SemiVariogram(tau, beta, f, lf, T, B, param.noise))
