# -*- coding: utf-8 -*-
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
        self.Ffun = Ffun
        self.DFfun = DFfun
        self.args = args
        self.kwargs = kwargs

        # Test the variable types.
        if not isinstance(w, np.ndarray)\
                or not isinstance(x0, np.ndarray):
            raise TypeError("Problem with variable type.")

        # Define input variables as row vectors.
        self.N = w.size
        self.w = w.reshape((self.N,))
        self.x = np.zeros(x0.shape)
        self.x[:] = x0[:]
        self.y = self.argmin_h_y(x0)
        self.K = self.x.size
        self.J = self.y.size

        # Test input variable consistency.
        aux = Ffun(self.x, *args, **kwargs)
        if not isinstance(aux, np.ndarray):
            raise TypeError("Problem with variable type of F output.")
        if aux.shape[0] != self.N or aux.shape[1] != self.J:
            raise ValueError("Problem with the definition of F.")

        aux = DFfun(self.x, self.y, *args, **kwargs)
        if not isinstance(aux, np.ndarray):
            raise TypeError("Problem with variable type of DF output.")
        if (aux.shape[0] != self.N or aux.shape[1] != self.K):
            raise ValueError("Problem with the definition of DF.")

    def set_parameters_fromfile(self, filename):
        r"""Load parameters from a configuration file in Linux format.

        :param filename: Name of the configuration file
        :type filename: str
        """
        # Load parameters from a configuration file
        self.param.load(filename)
        # Update Ffun and DFfun is need
        self.update_Ffun()

    @property
    def params(self):
        return self.param

    @params.setter
    def params(self, myparam):
        self.param = deepcopy(myparam)
        # Update Ffun and DFfun is need
        self.update_Ffun()

    def update_Ffun(self):
        r"""Redefine Ffun and DFfun if the scalar parameter alpha is strictly
        greater than 0 (i.e. there is a quadratic regularization on y).
        """
        if self.param.alpha > 0:
            Ffun_old = self.Ffun
            self.Ffun =\
                lambda x: np.concatenate((
                    Ffun_old(x, *self.args, **self.kwargs),
                    np.sqrt(self.param.alpha) * np.eye(self.J)),
                    axis=0)

            DFfun_old = self.DFfun
            self.DFfun =\
                lambda x, y: np.concatenate((
                    DFfun_old(x, y, *self.args, **self.kwargs),
                    np.zeros(self.K, self.K)),
                    axis=0)
            self.w = np.concatenate((self.w, np.zeros(self.N)))

    def Ffun_v(self, x, y, *args, **kwargs):
        return self.Ffun(x, *args, **kwargs) @ y

    # def DFfun_v(self, x, y, *args, **kwargs):
    #     return np.swapaxes(self.DFfun(x, *args, **kwargs), 1, 2) @ y

    def val_res(self, x):
        r"""Compute the residuals :math:`\epsilon_n` in :eq:`residuals`.

        :param x: Point where to compute the residuals
        :type x: :class:`numpy.ndarray` of size (N,)

        :return: Value of the residuals at the point given in argument
        """
        return self.Ffun_v(x, self.y, *self.args, **self.kwargs) - self.w

    def jac_res_x(self, x):
        r"""Compute the Jacobian of residuals with respect to :math:`x`.

        :param x: Point where to compute the Jacobian of the residuals
        :type x: :class:`numpy.ndarray` of size (N,)

        :return: Value of the Jacobian of residuals
        at the current point :math:`x`.
        """
        return self.DFfun(x, self.y, *self.args, **self.kwargs)

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
            h = h + self.param.reg.weight * tv(self.x) / self.K
        return h

    def argmin_h_x(self, param):
        r"""Minimize :math:`h` with respect to :math:`x`.

        :param param: Parameter for the algorithm
        :type param: :class:Varprox_Param

        :return: Minimizer of :math:`h` with respect to :math:`x`
        """
        ret_x = None
        # Minimizing h over x
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

        :param x_init: Point where to evaluate :math:`F`
        :type x_init: :class:`numpy.ndarray` of size (N,)

        :return: Minimizer of :math:`h` with respect to :math:`y`

        .. note::
            This operation corresponds to eq:`varpro`, which is the
            variable projection.
        """
        res = lsq_linear(self.Ffun(x, *self.args, **self.kwargs), self.w,
                         bounds=self.param.bounds_y)
        self.y = res.x
        return res.x

    def argmin_h(self):
        r"""Minimize :math:`h` with respect to :math:`(x, y)`.

        :param param: Parameter for the minimization of h wrt x
        :type param: :class:Solver_Param

        :return: Couple :math:`(x, y)` that minimize :math:`h`
        """
        h = np.finfo(float).max
        xtmp = np.zeros(self.x.shape)
        ytmp = np.zeros(self.y.shape)
        for it in range(self.param.maxit):
            xtmp[:] = self.x[:]
            ytmp[:] = self.y[:]
            self.x = self.argmin_h_x(self.param.solver_param)
            self.y = self.argmin_h_y(self.x)

            h0 = h
            h = self.h_value()
            if h0 != 0:
                if self.param.reg.name is None:
                    dh = (h0 - h) / h0 * 100
                    sdh = 1
                else:
                    dh = h0 - h
                    sdh = np.sign(dh)
                    dh = abs(dh) / h0 * 100
            else:
                dh = 0

            if self.param.verbose:
                print('varprox reg = {} | iter {:4d} / {}: cost = {:.6e} '
                      'improved by {:3.4f} percent.'
                      .format(self.param.reg.name, it,
                              self.param.maxit, h, sdh * dh))

            if dh < self.param.gtol_h:
                if dh < 0:
                    self.x[:] = xtmp[:]
                    self.y[:] = ytmp[:]
                break
        return self.x, self.y

    def generate_discrete_grad_mat(self, n):
        r"""Generate the discrete gradient matrix, i.e. the matrix with 1 on its
        diagonal and -1 on its first sub-diagonal.

        :param n: Dimension of the generated matrix.
        :type n: int

        :return: The discrete gradient matrix.
        """
        D = np.zeros([n, n])
        i, j = np.indices(D.shape)
        D[i == j] = 1
        D[i == j + 1] = -1
        D[0, n - 1] = -1
        return D

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
        x = self.x              # Primal variable
        v = np.zeros(x.shape)   # Dual variable
        L = self.generate_discrete_grad_mat(self.K)  # Linear operator
        crit = np.Inf          # Initial value of the objective function

        jac_res_x = self.jac_res_x(x)
        tau = 1 / LA.norm(jac_res_x.transpose() @ jac_res_x)
        sigma = 0.99 / (tau * LA.norm(L)**2)
        sigmarw = self.param.reg.weight / (sigma * self.K)

        # Main loop
        for n in range(self.param.solver_param.maxit):
            # 1) Primal update
            p = x - tau * self.gradient_g(x) - sigma * L.transpose() @ v
            # Projection on [bounds_x[0] + EPS, bounds_x[1] - EPS]
            p[p <= self.param.bounds_x[0]] = self.param.bounds_x[0] + EPS
            p[p >= self.param.bounds_x[1]] = self.param.bounds_x[1] - EPS
            # 2) Dual update
            vtemp = v + L @ (2 * p - x)
            q = vtemp - prox_l1(vtemp, sigmarw)
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
#                           Auxiliary Functions                                #
# ============================================================================ #
def tv(x):
    r"""
    This function computes the 1-dimensional discrete total variation of its
    input vector

    .. math::

        TV(x) = \sum_{n=1}^{N-1} x_{n+1} - x_{n}.

    :param x: input vector of length :math:`N`.
    :type x: :class:`numpy.ndarray` of size (N,)

    :return: 1-dimensional discrete total variation of the vector :math:`x`.
    """
    return np.sum(np.abs(np.diff(x)))


def prox_l1(data, reg_param):
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
