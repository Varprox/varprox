# -*- coding: utf-8 -*-
r"""
Tools for minimizing the penalized SNLS criterion.
"""
import numpy as np
from scipy.optimize import lsq_linear, least_squares
from dataclasses import dataclass
from numpy import linalg as LA

# ============================== CLASS MINIMIZE ============================= #
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
        if not isinstance(w, np.ndarray)\
                or not isinstance(x0, np.ndarray)\
                or not isinstance(y0, np.ndarray):
            raise TypeError("Problem with variable type.")

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
            raise TypeError("Problem with variable type of F output.")
        if aux.shape[0] != N or aux.shape[1] != J:
            raise ValueError("Problem with the definition of F.")

        aux = DFfun(self.x0, *args, **kwargs)
        if not isinstance(aux, np.ndarray):
            raise TypeError("Problem with variable type of DF output.")
        if (aux.shape[0] != N or aux.shape[1] != J or aux.shape[2] != K):
            raise ValueError("Problem with the definition of DF.")

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

        :param x: Point where to compute the residuals
        :type x: :class:`numpy.ndarray` of size (N,)

        :return: Value of the residuals at the point given in argument
        """
        return self.Ffun(x, *self.args, **self.kwargs) @ self.y - self.w

    def jac_res_x(self, x):
        r"""Compute the Jacobian of residuals with respect to :math:`x`.

        :param x: Point where to compute the Jacobian of the residuals
        :type x: :class:`numpy.ndarray` of size (N,)

        :return: Value of the Jacobian of residuals at the current point :math:`x`.
        """
        DF = self.DFfun(x, *self.args, **self.kwargs)
        eps_jac_x = np.zeros((DF.shape[0], x.size))
        for n in range(DF.shape[0]):
            eps_jac_x[n, :] = DF[n].T @ self.y
        return eps_jac_x

    def gradient_g(self, x):
        r"""Compute the gradient of the function :math:`g`.
        """
        return self.jac_res_x(x).transpose() @ self.val_res(x)

    def h_value(self):
        r"""Compute the value of the criterion :math:`h` in :eq:`criterion`
        using Equation :eq:`criterion2`.

        :return: Value of :math:`h` at the current point :math:`x`.
        """
        return np.sum(np.power(self.val_res(self.x), 2)) / 2

    def argmin_h_x(self, x_init, param):
        r"""Minimize :math:`h` with respect to :math:`x`.

        :param x_init: Initial point for the minimization algorithm
        :type x_init: :class:`numpy.ndarray` of size (N,)

        :param param: Parameter for the algorithm
        :type param: :class:RFBPD_Param

        :return: Minimizer of :math:`h` with respect to :math:`x`
        """
        ret_x = None
        # Minimizing h over x
        if param.reg is None:
            res = least_squares(fun=self.val_res, x0=x_init,
                                jac=self.jac_res_x,
                                bounds=self.bounds_x,
                                method='trf',
                                verbose=0,
                                gtol=param.gtol,
                                max_nfev=param.maxit
                                )
            ret_x = res.x
        elif param.reg == 'tv-1d':
            myparams = RFBPD_Param(param.reg_param)
            ret_x = self.rfbpd(x_init, myparams)
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
                         bounds=self.bounds_y)
        self.y = res.x
        return res.x

    def argmin_h(self, param):
        r"""Minimize :math:`h` with respect to :math:`(x, y)`.

        :param param: Parameter for the algorithm
        :type param: :class:Varprox_Param

        :return: Couple :math:`(x, y)` that minimize :math:`h`
        """
        h = self.h_value()
        x0 = np.zeros(self.x.shape)
        y0 = np.zeros(self.y.shape)
        for it in range(param.maxit):
            x0[:] = self.x[:]
            y0[:] = self.y[:]
            self.x = self.argmin_h_x(self.x, param)
            self.y = self.argmin_h_y(self.x)

            h0 = h
            h = self.h_value()
            if h0 != 0:
                dh = abs(h0 - h) / h0 * 100
            else:
                dh = 0
            if param.verbose:
                print('iter {:3d} / {}: cost = {:.6e} improved by {:3.4f} percent.'\
                      .format(it, param.maxit, h, dh))

            if dh < param.gtol:
                if dh < 0:
                    self.x[:] = x0[:]
                    self.y[:] = y0[:]
                break
        return (self.x, self.y)

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
        return D

    def rfbpd(self, x0, param):
        r"""Implementation of the rescaled Primal-dual Forward-backward
        algorithm (RFBPD)  to minimize the following optimization problem:

        .. math::
            :label: uncons_pb

            \min_{x\in\mathbb{R}^{n}} f(x) + g(Lx) + h(x) \, ,

        where :math:`f`, :math:`g`, and :math:`h` are proper, lower
        semi-continuous, and convex functions, :math:`h` is gradient
        :math:`\gamma`-Lipschitz, and :math:`L` is a linear operator from
        :math:`\mathbb{R}^{k}` to :math:`\mathbb{R}^{n}`.

        RFBPD iteration then reads:

        .. math::

            p_{n} &= \textrm{prox}_{\rho f} (x_{n}-\rho(\nabla h(x_{n})+\sigma L^{\top}x_{n}))\\
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

        :param x0: Initial value of the primal variable.
        :type x0: :class:`numpy.ndarray` (1-dimensional)

        :param param: Parameters of the algorithm.
        :type param: :class:`RFBPD_Param`

        :return: Final value of the primal variable.
        """
        # Constant for the projection on [EPS,1-EPS] corresponding to the
        # constraint that beta belongs to the open set ]0,1[
        EPS = 1e-8

        # Initialization
        n = x0.shape[0]         # Dimension of the ambient space
        x = x0                  # Primal variable
        v = np.zeros(x0.shape)  # Dual variable
        L = self.generate_discrete_grad_mat(n)  # Linear operator
        crit = np.Inf          # Initial value of the objective function

        param.tau = 1 / LA.norm(self.jac_res_x(x).transpose() @ self.jac_res_x(x))
        param.sigma = 1 / LA.norm(L)**2

        # Check the input parameters tau and sigma
        # beta = ??? # Value of the Lipschitz constant for the gradient of h
        # if 1/param.tau-param.sigma*LA.norm(L)**2 < beta/2:
        #     raise Exception("Input values for parameters tau and sigma are not valid.")
        # delta = 2-beta/(1/param.tau-param.sigma*LA.norm(L)**2)
        # if (delta < 1) or (delta >= 2):
        #     raise Exception("Input values for parameters tau and sigma are not valid.")

        # Main loop
        for n in range(param.max_iter):
            # 1) Primal update
            p = x - param.tau * self.gradient_g(x) -\
                param.sigma * L.transpose() @ v
            print(p)
            # Projection on [EPS,1-EPS]
            p[p <= 0] = EPS
            p[p >= 1] = 1 - EPS
            # 2) Dual update
            q = v + L @ (2 * p - x) - prox_l1(v + L @ (2 * p - x),
                                              param.reg_param / param.sigma)
            # 3) Inertial update
            LAMB = 1.2
            x = x + LAMB * (p - x)
            v = v + LAMB * (q - v)
            # 4) Check stopping criterion (convergence in term objective function)
            crit_old = crit
            crit = 0.5 * LA.norm(self.val_res(x))**2 + tv(x)
            if np.abs(crit_old - crit) < param.tol*crit:
                break
            # dh = (crit_old - crit) / crit
            # if np.abs(dh) < param.tol:
            #     break
            # else:
            #     print('sub iter {:3d} / {}: cost = {:.6e} improved by {:3.4f} percent.'
            #           .format(n, param.max_iter, crit, dh))

        return x

    def admm(self, x0, param):
        r"""Implementation of ADMM to minimize the optimization problem
        :eq:`uncons_pb`.

        ADMM iteration then reads:

        .. math::

            x_{n} &= \textrm{argmin}_{x \in \mathbb{R}^{n}} \frac{1}{2}\|Lx-y_{n}+z_{n}\|^{2} + \frac{1}{\gamma}(f(x)+h(x))\\
            s_{n} &= L x_{n}\\
            y_{n+1} &= \textrm{prox}_{g/\sigma} (z_{n}+s_{n})\\
            z_{n+1} &= z_{n}+s_{n}-y_{n+1}

        where :math:`\gamma` is the (strictly positive) parameter of the augmented
        Lagrangian, :math:`\gamma z_{n}` is the dual variable, and :math:`x_{n}`
        is the primal variable.

        In this implementation, :math:`f` is the indicator function of the set
        :math:`[\epsilon,1-\epsilon]^n`, :math:`g` is the :math:`\ell_{1}`-norm
        multiplied by a (strictly positive) regularization parameter, :math:`L`
        is the discrete gradient operator, and :math:`h` is the nonlinear
        least-squares.

        :param x0: Initial value of the primal variable.
        :type x0: :class:`numpy.ndarray` (1-dimensional)

        :param param: Parameters of the algorithm.
        :type param: :class:`ADMM_Param`

        :return: Final value of the primal variable.
        """
        # Constant for the projection on [EPS,1-EPS] corresponding to the
        # constraint that beta belongs to the open set ]0,1[
        EPS = 1e-8

        # Initialization
        n = x0.shape[0]         # Dimension of the ambient space
        x = x0                  # Primal variable
        y = np.zeros(x0.shape)  # Second primal variable
        z = np.zeros(x0.shape)  # Dual variable
        L = self.generate_discrete_grad_mat(n)  # Linear operator
        crit = np.Inf           # Initial value of the objective function

        # Main loop
        for n in range(param.max_iter):
            # 1) Minimize the augmented Lagrangian in x using a Forward-Backward
            #    subroutine
            for m in range(10000):
                # a) Forward step (gradient descent)
                x = x - L.transpose() @ (L @ x - y + z)\
                    - param.gamma * self.jac_res_x(x).transpose() @ self.val_res(x)
                # b) Backward step (projection on [EPS,1-EPS])
                x[x <= 0] = EPS
                x[x >= 1] = 1 - EPS
            # 2) Update temporary variable s
            s = L@x
            # 3) Minimize the augmented Lagrangian in y
            y = prox_l1(z + s, param.reg_param / param.gamma)
            # 4) Update the dual variable using a gradient ascent
            z = z + s - y
            # 5) Check stopping criterion (convergence in term objective function)
            crit_old = crit
            crit = 0.5 * LA.norm(self.val_res(x))**2 + tv(x)
            if np.abs(crit_old - crit) < param.tol*crit:
                break

        return x
# ============================ END CLASS MINIMIZE  ========================== #


# ========================= Helping Functions/Classes ======================= #
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


@dataclass
class Varprox_Param:
    gtol: float = 1e-3
    maxit: int = 1000
    verbose: bool = True
    reg: str = None
    reg_param: float = 0


@dataclass
class RFBPD_Param:
    reg_param: float
    max_iter: int = 10000
    tol: float = 1e-3
    sigma: float = 1
    tau: float = 1


@dataclass
class ADMM_Param:
    reg_param: float
    max_iter: int = 10000
    tol: float = 1e-3
    gamma: float = 1
# ============================================================================ #
