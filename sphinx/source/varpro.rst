Variable projection method
==========================

When there is no penalization (:math:`\lambda_1 = \lambda_2 = 0`), this problem
amounts to minimizing over :math:`y` the function

.. math::
    :label: criterionb

    g(x) = h(x, y^\ast(x))

where :math:`x^{\ast}(y)` is a so-called variable projection given by

.. math::
    :label: varpro

    y^{\ast}(x) \in \arg \min_{y} h(x, y).

To be completed
