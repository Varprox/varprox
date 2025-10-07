#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
# Credits
# #######
#
# Copyright(c) 2025-2025
# ----------------------
#
# * Institut de Mathématiques de Marseille <https://www.i2m.univ-amu.fr/>
# * Université d'Aix-Marseille <http://www.univ-amu.fr/>
# * Centre National de la Recherche Scientifique <http://www.cnrs.fr/>
#
# Contributors
# ------------
#
# * `Arthur Marmin <mailto:arthur.marmin@univ-amu.fr>`_
# * `Frédéric Richard <mailto:frederic.richard@univ-amu.fr>`_
#
#
# * This module is part of the package Varprox.
#
# Licence
# -------
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ######### COPYRIGHT #########
r"""
Tools for minimizing the penalized SNLS criterion.
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import numpy as np
from scipy.optimize import lsq_linear, least_squares
from numpy import linalg as LA
from varprox._parameters import Parameters
from copy import deepcopy
from scipy.linalg import circulant

# ============================================================================ #


# ============================================================================ #
#                                 CLASS MINIMIZE                               #
# ============================================================================ #
class Minimize:
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

    def __init__(self, x0, w, Ffun, DFfun, *args, **kwargs):
        r"""Constructor method.
        :param x0: initial guess for :math:`x`.
        :type x0: :ref:`ndarray` with shape (K,)
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

        # Optimisation parameters.
        self.param = Parameters()

        # Definition of Ffun and DFfun.
        self.Ffun = lambda x: Ffun(x, *args, **kwargs)
        self.DFfun = lambda x, y: DFfun(x, y, *args, **kwargs)
        self.Ffun_v = lambda x, y: self.Ffun(x) @ y

        # Test the variable types.
        if not isinstance(w, np.ndarray)\
                or not isinstance(x0, np.ndarray):
            raise TypeError("Problem with variable type.")

        # Define input variables as row vectors.
        self.N = w.size
        self.w = w.reshape((self.N,))
        self.x = np.zeros(x0.shape)
        self.x[:] = x0[:]
        self.K = self.x.size

        self.tv = TV(self.K)

        # Test input variable consistency.
        aux = Ffun(self.x, *args, **kwargs)
        self.J = aux.shape[1]
        if not isinstance(aux, np.ndarray):
            raise TypeError("Problem with variable type of F output.")
        if aux.shape[0] != self.N:
            raise ValueError("Problem with the definition of F.")

        aux = DFfun(self.x, np.zeros((self.J,)), *args, **kwargs)
        if not isinstance(aux, np.ndarray):
            raise TypeError("Problem with variable type of DF output.")
        if (aux.shape[0] != self.N or aux.shape[1] != self.K):
            raise ValueError("Problem with the definition of DF.")

        # Update Ffun, DFfun, and TV if needed
        self.update_Ffun()
        self.update_tv()
        # Initialize y.
        self.y = self.argmin_h_y(x0)

    def set_parameters_fromfile(self, filename):
        r"""Load parameters from a configuration file in Linux format.

        :param filename: Name of the configuration file
        :type filename: str

        :meta private:
        """
        # Load parameters from a configuration file
        self.param.load(filename)
        # Update Ffun and DFfun if needed
        self.update_Ffun()
        # Update TV if needed
        self.update_tv()

    @property
    def params(self):
        return self.param

    @params.setter
    def params(self, myparam):
        self.param = deepcopy(myparam)
        # Update Ffun and DFfun if needed
        self.update_Ffun()
        # Update TV if needed
        self.update_tv()

    def update_Ffun(self):
        r"""Redefine Ffun and DFfun if the scalar parameter alpha is strictly
        greater than 0 (i.e. there is a quadratic regularization on y).

        :meta private:
        """
        if self.param.alpha > 0 and self.J > 1:
            # solution provisoire.
            d = np.zeros(self.J)
            d[0] = 1
            d[1] = -1
            M = circulant(d)
            D = np.eye(self.J)
            for j in range(self.param.reg.order):
                D = D * M
            D = np.sqrt(self.param.alpha / self.J) * D

            Ffun_old = self.Ffun
            self.Ffun = lambda x: np.concatenate((Ffun_old(x), D), axis=0)

            DFfun_old = self.DFfun
            self.DFfun =\
                lambda x, y: np.concatenate((
                    DFfun_old(x, y), np.zeros((self.J, self.K))), axis=0)
            self.w = np.concatenate((self.w, np.zeros(self.J)))
            self.y = self.argmin_h_y(self.x)

    def update_tv(self):
        r"""Update the TV.

        :meta private:
        """
        self.tv = TV(self.K, self.param.reg.order)

    def val_res(self, x):
        r"""Compute the residuals :math:`\epsilon_n` in :eq:`residuals`.

        :param x: Point where to compute the residuals
        :type x: :class:`numpy.ndarray` of size (N,)

        :return: Value of the residuals at the point given in argument
        """
        return self.Ffun_v(x, self.y) - self.w

    def jac_res_x(self, x):
        r"""Compute the Jacobian of residuals with respect to :math:`x`.

        :param x: Point where to compute the Jacobian of the residuals
        :type x: :class:`numpy.ndarray` of size (N,)

        :return: Value of the Jacobian of residuals
            at the current point :math:`x`.
        """
        return self.DFfun(x, self.y)

    def gradient_g(self, x):
        r"""Compute the gradient of the function :math:`g`.
        """
        return self.jac_res_x(x).transpose() @ self.val_res(x) / self.N

    def h_value(self):
        r"""Compute the value of the criterion :math:`h` in :eq:`criterion`
        using Equation :eq:`criterion2`.

        :return: Value of :math:`h` at the current point :math:`x`.
        """
        h = np.mean(np.power(self.val_res(self.x), 2)) / 2

        if self.param.reg.name == 'tv-1d':
            h = h + self.param.reg.weight * self.tv.value(self.x) / self.K
        return h

    def argmin_h_x(self, param):
        r"""Minimize :math:`h` with respect to :math:`x`.

        :param param: Parameters for the algorithm
        :type param: Parameters

        :return: Minimizer of :math:`h` with respect to :math:`x`
        """
        ret_x = None
        # Minimizing h over x.
        if self.param.reg.name is None:
            res = least_squares(fun=self.val_res, x0=self.x,
                                jac=self.jac_res_x,
                                bounds=self.param.bounds_x,
                                method='trf',
                                verbose=0,
                                gtol=param.gtol,
                                max_nfev=param.maxit
                                )
            ret_x = res.x
        elif self.param.reg.name == 'tv-1d':
            ret_x = self.rfbpd()
        else:
            raise ValueError('The value of the parameter <reg> is unknown.')
        return ret_x

    def argmin_h_y(self, x):
        r"""Minimize :math:`h` with respect to :math:`y`.

        :param x: Point where to evaluate :math:`F`
        :type x: :class:`numpy.ndarray` of size (N,)

        :return: Minimizer of :math:`h` with respect to :math:`y`

        .. note::
            This operation corresponds to the variable projection.
        """
        res = lsq_linear(self.Ffun(x), self.w,
                         bounds=self.param.bounds_y)
        self.y = res.x
        return res.x

    def argmin_h(self):
        r"""Minimize :math:`h` with respect to :math:`(x, y)`.

        :return: A couple :math:`(x, y)` that minimizes :math:`h`
        """
        h = self.h_value()  # np.finfo('float').max
        xmin = np.zeros(self.x.shape)
        ymin = np.zeros(self.y.shape)
        xmin[:] = self.x[:]
        ymin[:] = self.y[:]
        hmin = h
        for it in range(self.param.maxit):
            self.x = self.argmin_h_x(self.param.solver_param)
            self.y = self.argmin_h_y(self.x)

            h0 = h
            h = self.h_value()
            if h0 > 0:
                if self.param.reg.name is None:
                    dh = (h0 - h) / h0 * 100
                    sdh = 1
                else:
                    dh = hmin - h
                    sdh = np.sign(dh)
                    dh = abs(dh) / h0 * 100
                if h < hmin:
                    hmin = h
                    xmin[:] = self.x[:]
                    ymin[:] = self.y[:]
            else:
                dh = 0

            if self.param.verbose:
                print('varprox reg = {:6s} | iter {:4d} / {}: cost = {:.6e} '
                      'improved by {:3.4f} percent.'
                      .format(str(self.param.reg.name), it,
                              self.param.maxit, h, sdh * dh))

            if dh < self.param.gtol_h:
                break

        self.x[:] = xmin[:]
        self.y[:] = ymin[:]

        return self.x, self.y

    def rfbpd(self):
        r"""Implementation of the rescaled Primal-dual Forward-backward
        algorithm (RFBPD) to minimize the following optimization problem:

        .. math::
            :label: uncons_pb

            \min_{x\in\mathbb{R}^{n}} f(x) + g(Lx) + h(x) \, ,

        where :math:`f`, :math:`g`, and :math:`h` are proper, lower
        semi-continuous, and convex functions, :math:`h` is gradient
        :math:`\gamma`-Lipschitz, and :math:`L` is a linear operator from
        :math:`\mathbb{R}^{k}` to :math:`\mathbb{R}^{n}`.

        RFBPD iteration then reads:

        .. math::

            p_{n} &= \textrm{prox}_{\rho f} (x_{n}-\rho(\nabla h(x_{n})+\sigma L^{\top}v_{n}))\\
            q_{n} &= (\mathrm{Id}-\textrm{prox}_{\lambda g/\sigma}) (v_{n}+L(2p_{n}-x_{n})\\
            (x_{n+1},v_{n+1}) &= (x_{n},v_{n}) + \lambda_{n}((p_{n},q_{n})-(x_{n},v_{n}))

        where :math:`\rho` and :math:`\sigma` are step sizes (strictly positive)
        on the primal and the dual problem respectively, :math:`\lambda_{n}` are
        inertial parameters, and :math:`v_{n}` is the rescaled dual variable.

        In this implementation, :math:`f` is the indicator function of the set
        :math:`[\epsilon,1-\epsilon]^n`, :math:`g` is the :math:`\ell_{1}`-norm
        multiplied by a (strictly positive) regularization parameter, :math:`L`
        is the discrete gradient operator, and :math:`h` is the nonlinear
        least-squares.

        Note that :math:`\rho` and :math:`\sigma` need to satisfy the following
        inequality in order to guarantee the convergence of the sequence
        :math:`(x_{n})` to a solution to the optimization:
        :math:`\rho^{-1}-\sigma\|L\|_{*}^{2} \geq \gamma/2`.

        :param param: Parameters of the algorithm.
        :type param: :class:`Varprox_Param`

        :return: Final value of the primal variable.
        """
        # Constant for the projection on [EPS,1-EPS] corresponding to the
        # constraint that beta belongs to the open set ]0,1[
        EPS = 1e-8

        # Initialization
        x = self.x             # Primal variable
        v = np.zeros(x.shape)  # Dual variable
        L = self.tv.L          # Linear operator
        crit = np.inf          # Initial value of the objective function

        # Main loop
        for n in range(self.param.solver_param.maxit):
            if np.mod(n, self.param.solver_param.maxit) == 0:
                jac_res_x = self.jac_res_x(x)
                tau = self.K / LA.norm(jac_res_x.transpose() @ jac_res_x)
                sigma = 0.99 / (tau * LA.norm(L)**2)
                sigmarw = self.param.reg.weight / (sigma * self.K)

            # 1) Primal update
            p = x - tau * self.gradient_g(x) - sigma * L.transpose() @ v
            # Projection on [bounds_x[0] + EPS, bounds_x[1] - EPS]
            p[p <= self.param.bounds_x[0]] = self.param.bounds_x[0] + EPS
            p[p >= self.param.bounds_x[1]] = self.param.bounds_x[1] - EPS
            # 2) Dual update
            vtemp = v + L @ (2 * p - x)
            q = vtemp - self.tv.prox_l1(vtemp, sigmarw)
            # 3) Inertial update
            LAMB = 1.8
            x = x + LAMB * (p - x)
            v = v + LAMB * (q - v)
            # 4) Check stopping criterion (convergence in term objective function)
            crit_old = crit
            crit = self.h_value()
            if np.abs(crit_old - crit) < self.param.solver_param.gtol * crit:
                break

        return x

# ============================ END CLASS MINIMIZE  =========================== #


# ============================================================================ #
#                                    CLASS TV                                  #
# ============================================================================ #
class TV():
    def __init__(self, dim, order=1):
        r"""
        This class implements8 the total variation (TV) regularization.

        :param dim: Dimension of the space where to apply TV
        :type dim: int

        :param order: Order of the TV (order of the differential operator).
        :type order: int
        """

        self.order = order
        self.dim = dim
        self.generate_discrete_grad_mat()

    def generate_discrete_grad_mat(self):
        r"""Generate the discrete gradient matrix, i.e. the matrix with 1 on its
        diagonal and -1 on its first sub-diagonal.

        :return: The discrete gradient matrix.
        """
        self.L = np.eye(self.dim)

        D = np.zeros([self.dim, self.dim])
        i, j = np.indices(D.shape)
        D[i == j] = 1
        D[i == j + 1] = -1
        D[0, self.dim - 1] = -1

        for i in range(self.order):
            self.L = np.matmul(self.L, D)

    def value(self, x):
        r"""
        This function computes the 1-dimensional discrete total variation of its
        input vector

        .. math::

            TV(x) = \sum_{n=1}^{N-1} x_{n+1} - x_{n}.

        :param x: input vector of length :math:`N`.
        :type x: :class:`numpy.ndarray` of size (N,)

        :return: 1-dimensional discrete total variation of the vector :math:`x`.
        """
        return np.sum(np.abs(self.L @ x))

    def prox_l1(self, data, reg_param):
        r"""
        This function implements the proximal operator of the l1-norm
        (a.k.a. soft thresholding).

        :param data: input vector of length :math:`N`.
        :type data: :class:`numpy.ndarray` of size (N,)

        :param reg_param: parameter of the operator (strictly positive).
        :type reg_param: float

        :return: The proximal operator of the l1-norm evaluated at the given point.
        """
        tmp = abs(data) - reg_param
        tmp = (tmp + abs(tmp)) / 2
        y = np.sign(data) * tmp
        return y

# ============================================================================ #
