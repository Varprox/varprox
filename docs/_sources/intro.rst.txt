Introduction.
=============

Problem statement.
------------------

A Separable Nonlinear Least Square criterion (SNLS) is of the form

.. math::
    :label: criterion

    h(x, y) = \frac{1}{2} \sum_{n=1}^N (F_n(x) y - w_n)^2,

where

    - :math:`x` and :math:`y` stand for two vectors in
      :math:`\mathbb{R}^K` and :math:`\mathbb{R}^J`, respectively,

    - :math:`w_n` are some real data,

    - :math:`F_n` is a function mapping :math:`\mathbb{R}^K`
      into :math:`\mathbb{R}^J`.

This package deals with the minimization over :math:`(x, y)`
of a penalized SNLS criterion

.. math::
    :label: pcriterion

    p(x, y) = h(x, y)
        + \lambda_1 \: r_1(x) + \lambda_2 \: r_2(y),

where :math:`r_1` and :math:`r_2` are some penalization terms (potentially
non-smooth) weighted by two positive scalars :math:`\lambda_1`
and :math:`\lambda_2`, respectively.


Denoting the residuals by

.. math::
    :label: residuals

    \epsilon_n(x, y) = F_n(x) y - w_n.

the criterion can also be written as

.. math::
    :label: criterion2

    h(x, y) = \frac{1}{2} \sum_{n=1}^N (\epsilon_n(x, y))^2
     + \lambda_1 \: r_1(x) + \lambda_2 \: r_2(y).

In a complete matrix form, it can further be written as

.. math::
    :label: criterion3

    h(x, y) = \frac{1}{2} \vert \epsilon(x, y) \vert^2
     + \lambda_1 \: r_1(x) + \lambda_2 \: r_2(y).

where :math:`\vert \cdot \vert` is the Euclidean norm in :math:`\mathbb{R}^n`
and :math:`\epsilon(x, y)` is the vector formed by terms
:math:`\epsilon_n(x, y)`.

Model fitting
-------------

The NSLS problem may arise when the data is to be fitted by a model mixing linear and non linear parts. For instance, consider a situation where the data :math:`w = (w_n)_{n=1}^N` comes from the observation of a unidimensional signal at some time points :math:`(t_n)_{n=1}^N` of :math:`\mathbb{R}`. Assume that the data observation can be described by an additive model 

.. math::
	w_n = f(t_n; x, y) + \epsilon_n,

where variables :math:`\epsilon_n` stand for independent Gaussian noise and the function :math:`f` for a signal model depending on parameters :math:`x` and :math:`y`. Further assume that the model :math:`f` is of the form

.. math::
	f(t) = \sum_{j=1}^J y_j \varphi_j(t; x) 

for some function :math:`\varphi_j(\cdot; x)` depending on parameters :math:`x`. Then, fitting the observed data :math:`w` with the model :math:`f` can be done by minimizing a SNLS of the form :eq:`criterion` for 

.. math::
	F_n(x)_j = \varphi_j(t_n; x), j=1,\cdots,J.

Such a modeling is generic and occurs in many fields of mathematical engineering. In the gallery of examples, we present some of its applications to the statistical inference of parameters for stochastic processes.

Variable projection method
--------------------------

When there is no penalization (:math:`\lambda_1 = \lambda_2 = 0`), the optimization problem
amounts to minimizing over :math:`y` the function 

.. math::
    :label: criterionb

    g(x) = h(x, y^\ast(x))

where :math:`x^{\ast}(y)` given by

.. math::
    :label: varpro

    y^{\ast}(x) \in \arg \min_{y} h(x, y).
    
This is the so-called variable projection methode which was introduced in :cite:`Golub_1973` to reduce the minimization problem to the single variable :math:`x`.

This package includes an extension of the variable projection to deal with cases when there are penalization and constraints on :math:`x`. The implemented methods are precisely described in :cite:`Richard-2023-TPMS` and :cite:`Marmin-2025-InvProb`. It is based on a proximal dual approach, called varprox !
