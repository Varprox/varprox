# -*- coding: utf-8 -*-
r"""
Tools for computing the semi-variogram of an anisotropic fractional
Brownian field and applying the fitting method.
"""
import numpy as np
from scipy.linalg import circulant
from afbf import perfunction, tbfield
from varprox import Minimize, Varprox_Param
from dataclasses import dataclass


def BasisFunctions(fun, t):
    """Values of the basis functions of the periodic function perfun at
       positions phi.

       :param basis: an array where values of the mth basis function
           of the representation are on the mth column.
       :type basis: :ref:`ndarray`
    """
    fun.Evaluate(t)
    return fun.basis.T


def SemiVariogram(tau, beta, f, T, B, noise=1):
    """Compute the semi-variogram of an AFBF or its increment field.
    """
    return Ffun(beta, f, None, T, B, noise) @ tau


def Ffun(beta, f, lf, T, B, noise=1, alpha=0):
    # Semi-variogram of the field
    F = np.concatenate((np.ones((f.shape[0], noise)),
                        0.5 * np.power(f, B @ beta) @ T), axis=1)
    if alpha > 0 and T.shape[1] > 1:
        c = np.zeros(T.shape[1])
        c[0] = 1
        c[1] = -1
        D = alpha * circulant(c).T
        D = np.concatenate((np.zeros((D.shape[0], noise)), D), axis=1)
        F = np.concatenate((F, D), axis=0)

    return F


def DFfun(beta, f, lf, T, B, noise=1, alpha=0):

    if alpha > 0 and T.shape[1] > 1:
        DF = np.zeros((f.shape[0] + T.shape[1], T.shape[1] + noise, B.shape[1]))
    else:
        DF = np.zeros((f.shape[0], T.shape[1] + noise, B.shape[1]))

    v = 0.5 * lf * np.power(f, B @ beta)
    for j in range(noise, DF.shape[1]):
        for k in range(DF.shape[2]):
            DF[0:f.shape[0], j, k] = v @ (T[:, j - noise] * B[:, k])

    return DF


def FitVariogram(model, lags, w, param):
    """Fit the field variogram using a coarse-to-fine multigrid strategy.
    """
    if model.hurst.ftype != "step" or model.topo.ftype != "step":
        raise ValueError("FitVariogram: only runs for step functions.")

    # Number of model parameters
    npar0_tau = model.topo.fparam.size
    npar0_beta = model.hurst.fparam.size

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

    bounds_beta = (0, 1)
    bounds_tau = (0, np.inf)

    w1 = w
    if param.multigrid:
        # Initialization
        hurst = perfunction(model.hurst.ftype, param=1)
        topo = perfunction(model.topo.ftype, param=1)
        B = BasisFunctions(hurst, phi)
        T = BasisFunctions(topo, phi) * dphi
        h = np.inf
        beta = np.array([0.5])
        tau = np.ones((param.noise + 1,))
        pb = Minimize(beta, tau, w1, Ffun, DFfun, None,
                      f, lf, T, B, param.noise)
        pb.param.bounds_x = bounds_beta
        pb.param.bounds_y = bounds_tau
        for i in range(1, 9):
            beta = np.array([i / 10])
            tau = pb.argmin_h_y(beta)
            h0 = pb.h_value()
            if h0 < h:
                h = h0
                beta2 = beta
                tau2 = tau
            npar_tau = 1
            npar_beta = 1
    else:
        npar_tau = npar0_tau
        npar_beta = npar0_beta
        beta2 = np.zeros((model.hurst.fparam.size,))
        beta2[:] = model.hurst.fparam[0, :]
        tau2 = np.zeros((model.topo.fparam.size,))
        tau2[:] = model.topo.fparam[0, :]
        if param.noise == 1:
            tau2 = np.insert(tau2, 0, 0)

    stop = False
    while stop is False:
        hurst = perfunction(model.hurst.ftype, param=npar_beta)
        topo = perfunction(model.topo.ftype, param=npar_tau)
        if param.multigrid:
            # Definition of the interval bounds for a step function (Hurst)
            ninter = hurst.finter.size
            Iv = np.linspace(-np.pi / 2, np.pi / 2, ninter + 1, True)[1:]
            hurst.ChangeParameters(hurst.fparam, Iv)
            # Definition of the interval bounds for a step function (topothesy)
            ninter = topo.finter.size
            Iv = np.linspace(-np.pi / 2, np.pi / 2, ninter + 1, True)[1:]
            topo.ChangeParameters(topo.fparam, Iv)
        else:
            hurst.fparam[:] = model.hurst.fparam[:]
            hurst.finter[:] = model.hurst.finter[:]
            topo.fparam[:] = model.topo.fparam[:]
            topo.finter[:] = model.topo.finter[:]

        B = BasisFunctions(hurst, phi)
        T = BasisFunctions(topo, phi) * dphi

        beta = beta2
        tau = tau2

        if param.verbose:
            print("Nb param: Hurst={:d}, Topo={:d}".format(
                hurst.fparam.size, topo.fparam.size))
            print("Tol = {:.5e}, Nepochs = {:d}".format(param.gtol, param.maxit))
        if param.alpha > 0 and T.shape[1] > 1:
            w1 = np.concatenate((w, np.zeros((T.shape[1],))), axis=0)
        pb = Minimize(beta, tau, w1, Ffun, DFfun,
                      bounds_beta, bounds_tau, f, lf, T, B, param.noise,
                      param.alpha)
        pb = Minimize(beta, tau, w1, Ffun, DFfun, None,
                      f, lf, T, B, param.noise)
        pb.param.bounds_x = bounds_beta
        pb.param.bounds_y = bounds_tau
        if beta.size > param.threshold_reg:
            myoptim_param = Varprox_Param(param.gtol, param.maxit,
                                          param.verbose, reg="tv-1d",
                                          reg_param=param.reg_param)
        else:
            myoptim_param = Solver_Param()

        beta, tau = pb.argmin_h(myoptim_param)

        stop = True
        if param.multigrid:
            npar0 = npar_tau
            npar = npar_tau * 2
            if npar <= npar0_tau:
                stop = False
                npar_tau = npar
                tau2 = np.zeros(npar)
                for k in range(npar0):
                    k2 = 2 * k
                    tau2[k2:k2 + 2] = tau[param.noise + k]
                if param.noise == 1:
                    tau2 = np.insert(tau2, 0, tau[0])

            npar0 = npar_beta
            npar = npar_beta * 2
            if npar <= npar0_beta:
                stop = False
                npar_beta = npar
                beta2 = np.zeros(npar)
                for k in range(npar0):
                    k2 = 2 * k
                    beta2[k2: k2 + 2] = beta[k]

        hurst.fparam[0, :] = beta[:]
        topo.fparam[0, :] = tau[param.noise:]
        emodel = tbfield("Estimated model", topo, hurst, model.tb)
        if param.noise == 1:
            emodel.noise = tau[0]
        else:
            emodel.noise = 0

    return (emodel, SemiVariogram(tau, beta, f, T, B, param.noise))


@dataclass
class Fit_Param:
    noise: int = 1
    k: np.ndarray = None
    multigrid: bool = True
    maxit: int = 1000
    gtol: float = 1e-6
    verbose: bool = True
    reg_param: float = 1
    alpha: float = 0
    threshold_reg: int = np.Inf
