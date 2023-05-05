# -*- coding: utf-8 -*-
r"""
Tools for minimizing the penalized SNLS criterion.
"""
import numpy as np
from scipy.optimize import lsq_linear, least_squares
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
import prox_tv as ptv
from dataclasses import dataclass
from numpy import linalg as LA

class minimize:
    r"""
    This class contains methods to minimize of a separable non-linear
    least square criterion, which is of the form:

    .. math::

        h(x, y) = \frac{1}{2} \sum_{n=1}^N \left(\epsilon_n(x, y))^2
        \quad \mathrm{with} \quad \epsilon_n(x,y) = F_n(x) y - w \right.

    :param x: first variable :math:`x` of the criterion :math:`h`.
    :type x: :class:`numpy.ndarray` of size (K,)

    :param y: second variable :math:`y` of the criterion :math:`h`.
    :type y: :class:`numpy.ndarray` of size (J,)

    :param w: set of observations :math:`(w_n)_n`.
    :type w: :class:`numpy.ndarray` of size (N,)

    :param Ffun: function which computes the mappings :math:`F_n` of
        the criterion, with the signature ``Ffun(x, *args, **kwargs)``.

        The argument x passed to this function F is an array of
        size (K,). It must allocate and return an array of shape
        (N, J) whose nth row F[n, :] corresponds
        to the vector :math:`F_n(x)`.
    :type Ffun: callable

    :param DFfun: a function which defines jacobian matrices
        :math:`DF_n(x)` of mappings :math:`F_n`,
        with the signature ``DFfun(x, *args, **kwargs)``.

        The argument x passed to this function DF is an array of
        size (K,). It must allocate and return an array of shape
        (N, J, K) whose nth term DF[n] corresponds
        to the jacobian matrix :math:`DF_n(x)`
        of :math:`F_n(x)`. DF[n, j, k] is the partial derivative
        of the jth component :math:`F_n(x)_j` with respect to the
        kth variable :math:`x_k`.
    :type DFfun: callable

    :param F: current values of :math:`F_n`.
    :type F: :class:`numpy.ndarray` of size (N, J)

    :param DF: current jacobian matrices of :math:`F_n`.
    :type DF: :class:`numpy.ndarray` of size (N, J, K)

    :param eps: residuals of Equation :eq:`residuals`.
        eps[n] correspond to :math:`\epsilon_n(x, y)`.
    :type eps: :class:`numpy.ndarray` of size (N, 1)

    :param eps_jac_x: jacobian of residuals :math:`\epsilon_n` with respect
        to :math:`x`.
        eps_jac_x[n, k] is the partial derivative of :math:`\epsilon_n`
        with respect to :math:`x_k`.
    :type eps_jac_x: :class:`numpy.ndarray` of size (N, K)

    :param bounds_x: Lower and upper bounds on :math:`x`.
            Defaults to no bounds. Each array must match the size of x0
            or be a scalar; in the latter case a bound will be the same
            for all variables. Use np.inf with an appropriate sign to disable
            bounds on all or some variables.
    :type bounds_x: 2-tuple of array_like, optional

    :param bounds_y: Lower and upper bounds on :math:`y`.
    :type bounds_y: 2-tuple of array_like, optional

    :param args, kwargs: Additional arguments passed to Ffun and DFfun.
            Empty by default.
    :type args, kwargs: tuple and dict, optional
    """

    def __init__(self, x0, y0, w, Ffun, DFfun,
                 bounds_x=(-np.inf, np.inf),
                 bounds_y=(-np.inf, np.inf),
                 *args, **kwargs):
        r"""Constructor method.
        :param x0: initial guess for :math:`x`.
        :type x0: :ref:`ndarray` with shape (K,)
        :param y0: initial guess for :math:`y`.
        :type y0: :ref:`ndarray` with shape (J,)
        :param w: vector :math:`w`.
        :type w: :ref:`ndarray` with shape (N,)
        :param Ffun: function to define the mapping :math:`F`.
        :param DFfun: function to define the jacobian of the mapping :math:`F`.
        :param bounds_x: Lower and upper bounds on :math:`x`.
        :type bounds_x: 2-tuple of array_like, optional
        :param bounds_y: Lower and upper bounds on :math:`y`.
        :type bounds_y: 2-tuple of array_like, optional
        :param args, kwargs: Additional arguments passed to Ffun and DFfun.
            Empty by default.
        :type args, kwargs: tuple and dict, optional
        """
        # Test the variable types.
        if (not isinstance(w, np.ndarray)
           or not isinstance(x0, np.ndarray)
           or not isinstance(y0, np.ndarray)):
            print('Problem with variable type.')
            return(None)

        # Define input variables as row vectors.
        N = w.size
        K = x0.size
        J = y0.size
        self.x0 = x0
        self.y0 = y0.reshape((J,))
        self.w = w.reshape((N,))

        # Test input variable consistency.
        aux = Ffun(self.x0, *args, **kwargs)
        if not isinstance(aux, np.ndarray):
            print('Problem with variable type of F output.')
            return(None)
        if aux.shape[0] != N or aux.shape[1] != J:
            print('Problem with the definition of F.')
            return(None)

        aux = DFfun(self.x0, *args, **kwargs)
        if not isinstance(aux, np.ndarray):
            print('Problem with variable type of DF output.')
            return(None)
        if (aux.shape[0] != N or aux.shape[1] != J or aux.shape[2] != K):
            print('Problem with the definition of DF.')
            return(None)

        self.Ffun = Ffun
        self.DFfun = DFfun
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y

        # Variable initializations.
        self.args = args
        self.kwargs = kwargs
        self.x = np.zeros(self.x0.shape)
        self.x[:] = self.x0[:]
        self.y = np.zeros(self.y0.shape)
        self.y[:] = self.y0[:]

    def val_res(self, x):
        r"""Compute the residuals :math:`\epsilon_n` in :eq:`residuals`.

        :return: update the attribute eps.
        """
        return(self.Ffun(x, *self.args, **self.kwargs) @ self.y - self.w)

    def jac_res_x(self, x):
        r"""Compute the jacobian of residuals with respect to :math:`x`.

        :return: update the attribute eps_jac_x.
        """
        DF = self.DFfun(x, *self.args, **self.kwargs)
        eps_jac_x = np.zeros((DF.shape[0], x.size))
        for n in range(DF.shape[0]):
            eps_jac_x[n, :] = DF[n].T @ self.y
        return(eps_jac_x)

    def h_value(self):
        r"""Compute the value of the criterion :math:`h` in :eq:`criterion`
        using Equation :eq:`criterion2`.

        :return: update the attributes h and eps.
        """
        return(np.sum(np.power(self.val_res(self.x), 2)) / 2)

    def argmin_h_x(self, x, gtol=1e-3, maxit=1000, reg=None, reg_param=0):
        r"""Minimize :math:`h` with respect to :math:`x`.

        :return: update the attribute x.
        """
        ret_x = None
        # Minimizing L over x
        if reg == None:
            res = least_squares(fun=self.val_res, x0=x,
                            jac=self.jac_res_x,
                            bounds=self.bounds_x,
                            method='trf',
                            verbose=0,
                            gtol=gtol,
                            max_nfev=maxit
                            )
            ret_x = res.x
        elif reg == 'tv-1d':
            myparams = RFBPD_Param(reg_param)
            ret_x = self.rfbpd(x, myparams)
        else:
            raise ValueError('The value of the parameter <reg> is unknown.')
        return ret_x

    def argmin_h_y(self, x):
        r"""Minimize :math:`h` with respect to :math:`y`.

        .. note::
            This operation corresponds to eq:`varpro`, which is the
            variable projection.
        """
        res = lsq_linear(self.Ffun(x, *self.args, **self.kwargs), self.w,
                         bounds=self.bounds_y)
        self.y = res.x
        return(res.x)

    def argmin_h(self, gtol=1e-3, maxit=1000, verbose=True, reg=None, reg_param=0):
        r"""Minimize :math:`h` with respect to :math:`(x, y)`.
        """
        h = self.h_value()
        x0 = np.zeros(self.x.shape)
        y0 = np.zeros(self.y.shape)
        for it in range(maxit):

            x0[:] = self.x[:]
            y0[:] = self.y[:]
            self.x = self.argmin_h_x(self.x, gtol, maxit, reg, reg_param)
            self.y = self.argmin_h_y(self.x)

            h0 = h
            h = self.h_value()
            if h0 != 0:
                dh = abs(h0 - h) / h0 * 100
            else:
                dh = 0
            if verbose:
                print('iter {:3d} / {}: cost = {:.6e} improved by {:3.4f} percent.'\
                      .format(it, maxit, h, dh))

            if dh < gtol:
                if dh < 0:
                    self.x[:] = x0[:]
                    self.y[:] = y0[:]
                break
        return(self.x, self.y)

    def generate_discrete_grad_mat(self, n):
        r"""Generate the discrete gradient matrix, i.e. the matrix with 1 on its
        diagonal and -1 on its first sub-diagonal.

        :param n: Dimension of the generated matrix.
        :type n: int

        :return: The discrete gradient matrix.     
        """
        D = np.zeros([n,n])
        i,j = np.indices(D.shape)
        D[i==j] = 1
        D[i==j+1] = -1
        return D

    def rfbpd(self, x0, param):
        r"""Implementation of the Primal-dual Forward-backward algorithm
        (rescaled version).
        
        :param x0: Initial value of the primal variable.
        :type x0: :class:`numpy.ndarray` (1-dimensional)

        :param param: Parameters of the algorithm.
        :type param: :class:`RFBPD_Param`

        :return: Final value of the primal variable.
        """
        # Define constants for the primal-dual Forward-Backward algorithm
        EPS = 1e-8

        # Initialization
        n = x0.shape[0]
        x = x0
        v = np.zeros(x0.shape)
        L = self.generate_discrete_grad_mat(n)
        crit = np.Inf

        #param.sigma = sigma=LA.norm(L)**(-2)

        # Main loop
        for n in range(param.max_iter):
            # 1) Primal update
            p = x - param.tau*self.jac_res_x(x).transpose()@self.val_res(x) - \
                param.sigma*L.transpose()@v
            # Projection on [EPS,1-EPS]
            p[p<=0] = EPS
            p[p>=1] = 1-EPS
            # 2) Dual update
            q = v + L@(2*p-x) - \
                ptv.tv1_1d(v + L@(2*p-x),
                           param.reg_param/param.sigma)
            # 3) Inertial update
            LAMB = 1
            x = x + LAMB*(p-x)
            v = v + LAMB*(q-v)
            # 4) Check stopping criterion (convergence in term objective function)
            crit_old = crit
            crit = 0.5*LA.norm(self.val_res(x))**2 + tv(x)
            if np.abs(crit_old - crit) < param.tol*crit:
                break
        return x

def tv(x):
    r"""
    This function compute the 1-dimensional discrete total variation of its
    input vector

    .. math::

        TV(x) = \sum_{n=1}^{N-1} x_{n+1} - x_{n}.

    :param x: input vector of length :math:`N`.
    :type x: :class:`numpy.ndarray` of size (N,)

    :return: 1-dimensional discrete total variation of the vector :math:`x`.
    """
    return np.sum(np.abs(np.diff(x)))

@dataclass
class RFBPD_Param:
    reg_param: float
    max_iter: int = 10000
    tol: float = 1e-3
    sigma: float = 1
    tau: float = 1
