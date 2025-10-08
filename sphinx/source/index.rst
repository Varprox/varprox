.. varprox documentation master file, created by
   sphinx-quickstart on Thu Dec  9 13:18:31 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to varprox's documentation!
===================================

.. .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.17154040.svg
..   :target: https://doi.org/10.5281/zenodo.17154040



The Package varprox is designed for solving penalized separable non-linear least squares problems. It extends the standard variable projection method by adding regularization on the non-linear variable.

Package features
================

- Methods for minimizing separable non-linear least squares problems with penalizations and box constraints on variables.

- Non linear model fitting in engineering.

- Applications to the statistical inference for stochastic processes. 


Installation from sources
=========================

The package source can be downloaded from the `repository <https://github.com/Varprox/varprox>`_. 

The package can be installed through PYPI with
 
 pip install varprox

Communication to the author
===========================

varprox is developed and maintained by Arthur Marmin and Frédéric Richard. For feed-back, contributions, bug reports, contact directly the `author <https://github.com/Varprox>`_, or use the `discussion <https://github.com/Varprox/varprox/discussions>`_ facility.


Licence
=======

varprox is under licence GNU GPL, version 3.


Citation
========

When using varprox, please cite the papers 
:cite:p:`Richard-2023-TPMS` and :cite:p:`Marmin-2025-InvProb`


.. .. image:: https://joss.theoj.org/papers/10.21105/joss.03821/status.svg
..   :target: https://doi.org/10.21105/joss.03821


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ./intro.rst
   ./varpro.rst
   ./auto_examples/index.rst
   ./api.rst
   ./biblio.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
