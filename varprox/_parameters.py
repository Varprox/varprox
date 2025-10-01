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
Implementation of the classes handling the parameters for the class Minimize
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
from configparser import ConfigParser
from dataclasses import dataclass
import numpy as np
# ============================================================================ #


# ============================================================================ #
#                                DATA STRUCTURES                               #
# ============================================================================ #
@dataclass
class SolverParam:
    gtol: float = 1e-3
    maxit: int = 1000


@dataclass
class RegParam:
    name: float = None
    weight: int = 0
    order: int = 1
# ============================================================================ #


# ============================================================================ #
#                                CLASS PARAMETERS                              #
# ============================================================================ #
class Parameters:
    def __init__(self, gtol=1e-4, maxit=1000, verbose=True, reg=RegParam(),
                 bounds_x=(-np.inf, np.inf), bounds_y=(-np.inf, np.inf),
                 solver_param=SolverParam()):
        self.gtol_h = gtol
        self.maxit = maxit
        self.verbose = verbose
        self.reg = reg
        self.alpha = 0
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.solver_param = solver_param

    def __repr__(self):
        mystr = "Object Parameters\n"
        mystr += "  gtol         = {:.3E}\n".format(self.gtol_h)
        mystr += "  maxit        = {:d}\n".format(self.maxit)
        mystr += "  verbose      = {}\n".format(self.verbose)
        mystr += "  reg          = Name: {} | Weight: {:.3E}\n"\
            .format(self.reg.name, self.reg.weight)
        mystr += "  alpha        = {:.3E}\n".format(self.alpha)
        mystr += "  bounds_x     = {}\n".format(self.bounds_x)
        mystr += "  bounds_y     = {}\n".format(self.bounds_y)
        mystr += "  solver param = Maxit: {} | Tol: {:.3E}\n"\
            .format(self.solver_param.maxit, self.solver_param.gtol)
        return mystr

    def load(self, filename):
        """Load the parameters from a file in the format of Linux configuration
        file.

        :param filename: Name of the file containing the parameters
        :type: str
        """
        parser = ConfigParser()
        parser.read(filename)

        self.gtol_h = parser.getfloat('general-param', 'gtol')
        self.maxit = parser.getint('general-param', 'maxit')
        self.verbose = parser.getboolean('general-param', 'verbose')
        self.reg = RegParam(parser.get('regul-param', 'reg_name'),
                            parser.getfloat('regul-param', 'reg_param'))
        self.alpha = parser.getfloat('regul-param', 'alpha')

        if parser.get('general-param', 'lbound_x') == '-inf':
            lower_bd_x = -np.inf
        else:
            lower_bd_x = parser.getfloat('general-param', 'lbound_x')
        if parser.get('general-param', 'ubound_x') == 'inf':
            upper_bd_x = np.inf
        else:
            upper_bd_x = parser.getfloat('general-param', 'ubound_x')
        if parser.get('general-param', 'lbound_y') == '-inf':
            lower_bd_y = -np.inf
        else:
            lower_bd_y = parser.getfloat('general-param', 'lbound_y')
        if parser.get('general-param', 'ubound_y') == 'inf':
            upper_bd_y = np.inf
        else:
            upper_bd_y = parser.getfloat('general-param', 'ubound_y')

        self.bounds_x = (lower_bd_x, upper_bd_x)
        self.bounds_y = (lower_bd_y, upper_bd_y)

        self.solver_param = SolverParam(parser.getfloat('solver-param', 'tol'),
                                        parser.getint('solver-param', 'maxit'))

    def save(self, filename):
        """Save the parameters to a file in the format of Linux configuration
        file.

        :param filename: Name of the output file
        :type: str
        """
        config = ConfigParser()

        config.add_section('general-param')
        config.set('general-param', 'maxit', str(self.maxit))
        config.set('general-param', 'gtol', str(self.gtol_h))
        config.set('general-param', 'verbose', str(self.verbose))
        if self.bounds_x[0] == -np.inf:
            config.set('general-param', 'lbound_x', '-inf')
        else:
            config.set('general-param', 'lbound_x', str(self.bounds_x[0]))
        if self.bounds_x[1] == np.inf:
            config.set('general-param', 'ubound_x', 'inf')
        else:
            config.set('general-param', 'ubound_x', str(self.bounds_x[1]))
        if self.bounds_y[0] == -np.inf:
            config.set('general-param', 'lbound_y', '-inf')
        else:
            config.set('general-param', 'lbound_y', str(self.bounds_y[0]))
        if self.bounds_y[1] == np.inf:
            config.set('general-param', 'ubound_y', 'inf')
        else:
            config.set('general-param', 'ubound_y', str(self.bounds_y[1]))

        config.add_section('regul-param')
        config.set('regul-param', 'reg_name', str(self.reg.name))
        config.set('regul-param', 'reg_param', str(self.reg.weight))
        config.set('regul-param', 'alpha', str(self.alpha))

        config.add_section('solver-param')
        config.set('solver-param', 'tol', str(self.solver_param.gtol))
        config.set('solver-param', 'maxit', str(self.solver_param.maxit))

        with open(filename, 'w') as f:
            config.write(f)
# ============================================================================ #
