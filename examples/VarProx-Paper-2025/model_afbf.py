# -*- coding: utf-8 -*-
r"""Tools for computing the semi-variogram of an anisotropic fractional
Brownian field and applying the fitting method.
"""
import numpy as np
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
    mw = np.mean(np.power(w, 2))
    reg_weight = param.reg.weight
    reg_alpha = param.alpha * mw
    val_ref = -1

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
        hurst.ChangeParameters(np.array([0.5]))
    else:
        hurst = perfunction(model.hurst.ftype, param=npar0_beta)
        topo = perfunction(model.topo.ftype, param=npar0_tau)
        hurst.ChangeParameters(model.hurst.fparam[:], model.hurst.finter[:])
        topo.ChangeParameters(model.topo.fparam[:], model.topo.finter[:])

    beta = np.zeros((hurst.fparam.size,))
    beta[:] = hurst.fparam[0, :]

    stop = False
    while stop is False:
        # Update the basis function for hurst and topothesy.
        B = BasisFunctions(hurst, phi)
        T = BasisFunctions(topo, phi) * dphi

        # Cancel the tv regularization when less than threshold_reg parameters
        # are involved.
        pb = Minimize(beta, w, Ffun, DFfun, f, lf, T, B, param.noise)
        if reg_name == "tv-1d":
            if beta.size < param.threshold_reg:
                param.reg.name = None
            else:
                if val_ref == -1:
                    val_ref = pb.h_value()
                    param.reg.weight = reg_weight * val_ref
                    param.reg.name = reg_name

        param.alpha = reg_alpha
        pb.params = param

        if param.verbose:
            print("Nb param: Hurst={:d}, Topo={:d}".format(
                hurst.fparam.size, topo.fparam.size))
            print("Tol = {:.5e}, Nepochs = {:d}".format(param.gtol_h,
                                                        param.maxit))

        beta, tau = pb.argmin_h()
        topo.ChangeParameters(tau[param.noise:], topo.finter)
        hurst.ChangeParameters(beta[:], hurst.finter)

        stop = True
        if param.multigrid:
            # Increase the number of parameters for the topothesy.
            npar = (tau.size - param.noise) * 2
            if npar <= npar0_tau:
                stop = False
                topo0 = topo
                topo = perfunction(model.topo.ftype, param=npar)
                Iv = np.linspace(-np.pi / 2, np.pi / 2, npar + 1, True)
                centers = (Iv[0:-1] + Iv[1:]) * 0.5
                topo0.Evaluate(centers)
                topo.ChangeParameters(topo0.values[0, :], Iv[1:])

            # Increase the number of parameters for the Hurst function.
            npar = beta.size * 2
            if npar <= npar0_beta:
                stop = False
                hurst0 = hurst
                hurst = perfunction(model.hurst.ftype, param=npar)
                Iv = np.linspace(-np.pi / 2, np.pi / 2, npar + 1, True)
                centers = (Iv[0:-1] + Iv[1:]) * 0.5
                hurst0.Evaluate(centers)
                hurst.ChangeParameters(hurst0.values[0, :], Iv[1:])
                beta = np.zeros((hurst.fparam.size,))
                beta[:] = hurst.fparam[0, :]

    emodel = tbfield("Estimated model", topo, hurst, model.tb)
    if param.noise == 1:
        emodel.noise = tau[0]
    else:
        emodel.noise = 0
    topo.fname = "topothesy"
    hurst.fname = "Hurst function"

    return (emodel, SemiVariogram(tau, beta, f, lf, T, B, param.noise))
