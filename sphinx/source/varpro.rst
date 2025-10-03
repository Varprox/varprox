Variable projection method
==========================

When there is no penalization (:math:`\lambda_1 = \lambda_2 = 0`), this problem
amounts to minimizing over :math:`y` the function 

.. math::
    :label: criterionb

    g(x) = h(x, y^\ast(x))

where :math:`x^{\ast}(y)` is a so-called variable projection :footcite:t:`Golub_1973` given by

.. math::
    :label: varpro

    y^{\ast}(x) \in \arg \min_{y} h(x, y).

The approach we implemented in this paper extends the variable projection to the case when there are penalization. The methods are described in 
