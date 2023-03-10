# -*- coding: utf-8 -*-
r"""
===========
Perceptron.
===========

This module contains tools to define a perceptron with an hidden layer.

Let :math:`v` be the input vector of size M. The input node :math:`v`
is sent to :math:`J` nodes :math:`z_j` of an hidden layer with some mappings

.. math::

    z_j = \varphi(\sum_{m}^M x_{jm} v_m)

depending on the set of parameters :math:`x =(x_{jm})_{jm}`
and an activation function :math:`\varphi`
(e.g :math:`\varphi(t) = \max(0, t)`).

The nodes of the hidden layer are then linearly combined to form
an ouput variable :math:`v`

.. math::

    w = \sum_{j=1}^J y_j z_j = \sum_{j=1}^J y_j \varphi(\sum_{m}^M x_{jm} v_m)

using some other parameters :math:`x = (x_j)_j`.

The relationship between the output and the input can be described with the
function :math:`P`

.. math::

    w = P(v; x, y) = \Phi(v; x) y

with

.. math::

    \Phi_j(v; y) = \varphi(\sum_{m}^M y_{jm} v_m).

Given a set of input and output pairs $(v_n, w_n)$, the perceptron can be
learned by minimizing the cost function

.. math::
    h(x, y) = \sum_{n=1}^N \left( F_n(x) y - w_n\right)

with :math:`F_n(x) = \Phi(v_n; x)`.

"""
import numpy as np
import scipy as sp
from numpy.matlib import repmat

from varprox import minimize


def HiddenLayer(x, v):
    """
    :param x: parameters of the perceptron.
    :type x: :ref:`ndarray` of size M x J, where J is the number of nodes on
        the hidden layer.
    :param v: input of the perceptron
    :type v: :ref:`ndarray` of size (N, M).

    """
    M = v.shape[1]  # size of input.
    J = int(x.size / M)  # number of nodes on the hidden layer.
    return(v @ x.reshape((M, J)))


def fmodel(x, y, v):
    return((Ffun(x, v) @ np.expand_dims(y, 1)).flatten())


def phi(h):
    """ReLU function.
    """
    return(np.maximum(0, h))


def dphi(h):
    """Derivative of the ReLU function.
    """
    h[h <= 0] = 0
    h[h > 0] = 1
    return(h)


# def phi(h):
#     """Logit function.
#     """
#     return(1 / (1 + np.exp(- h)))


# def dphi(h):
#     """Derivative of the Logit function.
#     """
#     return(1 / (1 + np.exp(h)))


def Ffun(x, v):
    """Define functions :math:`F_n`.
    """
    return(phi(HiddenLayer(x, v)))


def DFfun(x, v):
    """Define the jacobian of :math:`F_n`.
    """
    hh = dphi(Ffun(x, v))

    J = hh.shape[1]  # nb of nodes on the hidden layer.
    K = hh.shape[1]  # number of weights for each node.
    DF = np.zeros((hh.shape[0], J, x.size))

    for n in range(hh.shape[0]):
        P = np.expand_dims(hh[n, :], axis=1)
        Q = np.expand_dims(v[n, :], axis=0)
        Q = repmat(Q, 1, K)
        DF[n, :, :] = P @ Q
    return(DF)


# Parameters.
M = 5  # size of the input.
J = 10  # number of nodes on the hidden layer
N = 1000  # number of examples.
sig = 0.01  # noise std.

# Generate the model and the data.
x = np.random.randn(J * M,)  # weights of the hidden layers.
y = np.random.randn(J,)  # weights for the output.

# synthetic input.
v = np.random.rand(1, M) * J +\
    np.random.randn(N, M) @ np.diag(np.random.rand(M)) @\
    sp.linalg.orth(np.random.randn(M, M))
# synthetic output
f = fmodel(x, y, v)
w = f + sig * np.random.randn(N)


# Minimize the problem.
x0 = np.zeros(x.shape)
x0[:] = x[:]
y0 = np.zeros(y.shape)
y0[:] = y[:]

y = np.random.randn(y.size)
bounds = (-np.inf, np.inf)
pb = minimize(x0, y, w, Ffun, DFfun, bounds, bounds, v)
y1 = pb.argmin_h_y(x0)
print('argmin y')
print(y0)
print(y1)

x = np.random.randn(x.size)
pb = minimize(x, y0, w, Ffun, DFfun, bounds, bounds, v)
x1 = pb.argmin_h_x(x)
print('argmin x')
print(x0)
print(x1)

x = np.random.randn(x.size)
y = np.random.randn(y.size)
pb = minimize(x, y, w, Ffun, DFfun, bounds, bounds, v)
x1, y1 = pb.argmin_h()
print('argmin (x, y)')
# print(x0)
# print(x1)

# print(y0)
# print(y1)
print(pb.h_value() / (np.sum(np.power(w, 2)) / 2) * 100)
