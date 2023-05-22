# -*- coding: utf-8 -*-
r"""
Tools for computing the semi-variogram of an anisotropic fractional
Brownian field and applying the fitting method.
"""
import numpy as np
from afbfstatio import perfunction, tbfield
from varprox import minimize


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
    if len(f) == 1:
        # Semi-variogram of the field.
        return(SemiVariogram_AFBF(tau, beta, f[0], T, B, noise))
    else:
        # Semi-variogram of increment field.
        return(2 * (SemiVariogram_AFBF(tau, beta, f[0], T, B, noise)
                    + SemiVariogram_AFBF(tau, beta, f[1], T, B, noise))
               - SemiVariogram_AFBF(tau, beta, f[2], T, B, noise)
               - SemiVariogram_AFBF(tau, beta, f[3], T, B, noise))


def Ffun(beta, f, lf, T, B, noise=1):
    if len(f) == 1:
        # Semi-variogram of the field.
        return(Ffun_AFBF(beta, f[0], T, B, noise))
    else:
        # Semi-variogram of increment field.
        return(2 * (Ffun_AFBF(beta, f[0], T, B, noise)
                    + Ffun_AFBF(beta, f[1], T, B, noise))
               - Ffun_AFBF(beta, f[2], T, B, noise)
               - Ffun_AFBF(beta, f[3], T, B, noise))


def DFfun(beta, f, lf, T, B, noise=1):

    if len(f) == 1:
        # Semi-variogram of the field.
        return(DFfun_AFBF(beta, f[0], lf[0], T, B, noise))
    else:
        # Semi-variogram of increment field.
        return(2 * (DFfun_AFBF(beta, f[0], lf[0], T, B, noise)
                    + DFfun_AFBF(beta, f[1], lf[1], T, B, noise))
               - DFfun_AFBF(beta, f[2], lf[2], T, B, noise)
               - DFfun_AFBF(beta, f[3], lf[3], T, B, noise))


def SemiVariogram_AFBF(tau, beta, f, T, B, noise=1):
    """Compute the semi-variogram of an AFBF.
    """
    return Ffun_AFBF(beta, f, T, B, noise) @ tau


def Ffun_AFBF(beta, f, T, B, noise=1):
    return np.concatenate((np.ones((f.shape[0], noise)),
                           0.5 * np.power(f, B @ beta) @ T), axis=1)


def DFfun_AFBF(beta, f, lf, T, B, noise=1):
    DF = np.zeros((f.shape[0], T.shape[1] + noise, B.shape[1]))
    v = 0.5 * lf * np.power(f, B @ beta)
    for j in range(noise, DF.shape[1]):
        for k in range(DF.shape[2]):
            DF[:, j, k] = v @ (T[:, j - noise] * B[:, k])
    return DF


def FitVariogram(model, lags, w, noise=1,
                 multigrid=True, maxit=1000, gtol=1e-6, verbose=1):
    """Fit the field variogram using a coarse-to-fine multigrid strategy.
    """
    if model.hurst.ftype != "step" or model.topo.ftype != "step":
        print("FitVariogram: only runs for step functions.")
        return(0)

    # Number of model parameters.
    npar0_tau = model.topo.fparam.size
    npar0_beta = model.hurst.fparam.size

    # Ffun and Dfun parameters.
    # Turning-band angles.
    phi = np.zeros(model.tb.Kangle.shape)
    phi[:] = model.tb.Kangle[:]
    # dphi = np.diff(phi)
    phi = phi[1:]
    phi = np.expand_dims(phi, axis=0)
    # dphi = np.expand_dims(dphi, axis=1)
    # Coordinates where variograms are computed.
    xy = lags.xy
    # N = lags.N

    increm = model.kappa.fparam[0, 0]
    csphi = np.concatenate((np.cos(phi), np.sin(phi)), axis=0)

    # K = model.tb.Qangle.size - 1
    # csphi = np.concatenate((model.tb.Qangle[1:].reshape((1, K)),
    #                         model.tb.Pangle[1:].reshape((1, K))), axis=0)
    cxy = xy @ csphi
    # f = [np.power(cxy, 2) / N**2]
    f = [np.power(cxy, 2)]
    if increm is not None:
        # f.append(np.power(increm * np.ones(cxy.shape), 2) / N**2)
        # f.append(np.power(cxy - increm, 2) / N**2)
        # f.append(np.power(cxy + increm, 2) / N**2)
        f.append(np.power(increm * np.ones(cxy.shape), 2))
        f.append(np.power(cxy - increm, 2))
        f.append(np.power(cxy + increm, 2))

    lf = []
    for j in range(len(f)):
        lf.append(np.zeros(f[j].shape))
        ind = np.nonzero(f[j] > 0)
        lf[j][ind] = np.log(f[j][ind])

    bounds_beta = (0, 1)
    bounds_tau = (0, np.inf)

    if multigrid:
        # Initialization.
        hurst = perfunction(model.hurst.ftype, param=1)
        topo = perfunction(model.topo.ftype, param=1)
        B = BasisFunctions(hurst, phi)
        T = BasisFunctions(topo, phi)  # * dphi
        h = np.inf
        beta = np.array([0.5])
        tau = np.ones((noise + 1,))
        pb = minimize(beta, tau, w, Ffun, DFfun,
                      bounds_beta, bounds_tau, f, lf, T, B, noise)
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
        if noise == 1:
            tau2 = np.insert(tau2, 0, 0)

    stop = False
    while stop is False:
        hurst = perfunction(model.hurst.ftype, param=npar_beta)
        topo = perfunction(model.topo.ftype, param=npar_tau)
        if multigrid:
            # Definition of the interval bounds for a step function.
            ninter = hurst.finter.size
            Iv = np.linspace(-np.pi / 2, np.pi / 2, ninter + 1, True)[1:]
            hurst.ChangeParameters(hurst.fparam, Iv)
            # Definition of the interval bounds for a step function.
            ninter = topo.finter.size
            Iv = np.linspace(-np.pi / 2, np.pi / 2, ninter + 1, True)[1:]
            topo.ChangeParameters(topo.fparam, Iv)
        else:
            hurst.fparam[:] = model.hurst.fparam[:]
            hurst.finter[:] = model.hurst.finter[:]
            topo.fparam[:] = model.topo.fparam[:]
            topo.finter[:] = model.topo.finter[:]

        B = BasisFunctions(hurst, phi)
        T = BasisFunctions(topo, phi)  # * dphi

        beta = beta2
        tau = tau2

        if verbose > 0:
            print("Nb param: Hurst=%d, Topo=%d" %
                  (hurst.fparam.size, topo.fparam.size))
            print("Tol = %e, Nepochs = %d" % (gtol, maxit))
        pb = minimize(beta, tau, w, Ffun, DFfun,
                      bounds_beta, bounds_tau, f, lf, T, B, noise)
        beta, tau = pb.argmin_h(gtol, maxit, verbose)

        stop = True
        if multigrid:
            npar0 = npar_tau
            npar = npar_tau * 2
            if npar <= npar0_tau:
                stop = False
                npar_tau = npar
                tau2 = np.zeros(npar)
                for k in range(npar0):
                    k2 = 2 * k
                    tau2[k2:k2 + 2] = tau[noise + k]
                if noise == 1:
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

        print(tau)
        hurst.fparam[0, :] = beta[:]
        topo.fparam[0, :] = tau[noise:]
        emodel = tbfield("Estimated model", topo, hurst, None, model.tb)
        emodel.kappa.fparam[:] = increm
        if noise == 1:
            emodel.noise = tau[0]
        else:
            emodel.noise = 0

    return (emodel, SemiVariogram(tau, beta, f, T, B, noise))
