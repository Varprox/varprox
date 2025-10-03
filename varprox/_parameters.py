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
import pathlib
import yaml
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
    name: str = None
    weight: int = 0
    order: int = 1
# ============================================================================ #


# ============================================================================ #
#                                CLASS PARAMETERS                              #
# ============================================================================ #
class Parameters:
    r"""This class enables to handle parameters for the optimization.

    :param maxit: Maximum number of iterations.
    :type maxit: int. Default = 1000.

    :param gtol: Tolerance for the stopping critetion.
    :type gtol: float. Default = 1e-4.

    :param lbound_x: Lower bound for the non linear variable x.
    :type lbound_x: float. Default = -infty.

    :param ubound_x: Upper bound for the non linear variable x.
    :type ubound_x: float. Default = +infty.

    :param lbound_y: Lower bound for the linear variable y.
    :type lbound_y: float. Default = -infty.

    :param ubound_y: Lower bound for the linear variable y.
    :type ubound_y: float. Default = infty.

    :param verbose: Verbose if True.
    :type verbose: boolean.

    :param alpha: Regularization weight on the linear variable x.
    :type alpha: float.

    :param itermax_neg:
    :type itermax_neg: int

    :param reg_name:
        Type of regularization on the non linear variable x.
        (None = no regularization, "tv-1d" = TV regularization).
    :type reg_name: str.

    :param reg_param: Regularization weight on the non linear variable x.
    :type reg_param: float.

    :param order: order of the derivatives to be penalized.
    :type order: int.

    :param tol: Tolerance for the sub-optimization on the linear variable y.
    :type tol: float.

    :param maxit:
        Maximal number for the sub-optimization on the linear variable y.
    :type maxit: float.
    """

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
        """Load the parameters from a file.

        :param filename: Name of the file containing the parameters
        :type filename: str
        """
        file_ext = pathlib.Path(filename).suffix
        if file_ext == '.yml':
            self.load_yaml(filename)
        elif file_ext == '.ini':
            self.load_initfile(filename)
        else:
            raise ValueError("The configuration file format is unknown: "
                             "only YAML and Linux configuration file are "
                             "supported.")

    def save(self, filename, filetype):
        """Save the parameters.

        :param filename: Name of the output file
        :type filename: str

        :param filetype:
            NameFormat of the output file. Must be 'yaml' or 'init'.
        :type filetype: str
        """
        if filetype == 'yaml':
            self.save_yaml(filename)
        elif filetype == 'init':
            self.save_initfile(filename)
        else:
            raise ValueError("The configuration file format is unknown: "
                             "only YAML and Linux configuration file are "
                             "supported.")

    def load_initfile(self, filename):
        """Load the parameters from a file in the format of Linux configuration
        file.

        :param filename: Name of the file containing the parameters
        :type filename: str

        :meta private:
        """
        parser = ConfigParser()
        try:
            parser.read(filename)
        except:
            print("Fail to load " + filename + ".")

        self.gtol_h = parser.getfloat('general-param', 'gtol')
        self.maxit = parser.getint('general-param', 'maxit')
        self.verbose = parser.getboolean('general-param', 'verbose')

        regname = parser.get('regul-param', 'reg_name')
        if regname == "None":
            self.reg = RegParam(None,
                            parser.getfloat('regul-param', 'reg_param'))
        else:
            self.reg = RegParam(regname,
                                parser.getfloat('regul-param', 'reg_param'))
        self.alpha = parser.getfloat('general-param', 'alpha')

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

        if lower_bd_x >= upper_bd_x:
            raise ValueError("Upper bound on nonlinear variables must be "
                             "higher than its lower bound.")
        if lower_bd_y >= upper_bd_y:
            raise ValueError("Upper bound on linear variables must be "
                             "higher than its lower bound.")

        self.bounds_x = (lower_bd_x, upper_bd_x)
        self.bounds_y = (lower_bd_y, upper_bd_y)

        self.solver_param = SolverParam(parser.getfloat('solver-param', 'tol'),
                                        parser.getint('solver-param', 'maxit'))

    def load_yaml(self, filename):
        """Load the parameters from a YAML file.

        :param filename: Name of the file containing the parameters
        :type filename: str

        :meta private:
        """
        with open(filename, 'r') as f:
            try:
                myconfig = yaml.safe_load(f)
            except:
                print("Fail to load " + filename + ".")

        try:
            if not isinstance(myconfig['general-param']['maxit'], int):
                raise ValueError("Maximum number of iterations has to be an integer.")
            self.maxit = int(myconfig['general-param']['maxit'])
            
            if not isinstance(myconfig['general-param']['gtol'], float) \
               and not isinstance(myconfig['general-param']['gtol'], int):
                raise ValueError("Global tolerance has to be a float.")
            if myconfig['general-param']['gtol'] <= 0:
                raise ValueError("Global tolerance must be strictly positive.")
            self.gtol_h = float(myconfig['general-param']['gtol'])

            if not isinstance(myconfig['general-param']['verbose'], bool):
                raise ValueError("Verbose has to be a boolean.")
            self.verbose = myconfig['general-param']['verbose']

            if not isinstance(myconfig['regul-param']['reg_name'], str):
                raise ValueError("The regularization for the nonlinear "
                                 "variables has to be 'tv-1d' or 'None'.")
            if not isinstance(myconfig['regul-param']['reg_param'], float) \
               and not isinstance(myconfig['regul-param']['reg_param'], int):
                raise ValueError("The regularization parameter for nonlinear "
                                 "variables has to be a float.")
            if myconfig['regul-param']['reg_param'] < 0:
                raise ValueError("Regularization parameter must be positive.")
            regname = myconfig['regul-param']['reg_name']
            if regname == "None":
                RegParam(None,
                         myconfig['regul-param']['reg_param'])
            else:
                self.reg = RegParam(regname,
                                    myconfig['regul-param']['reg_param'])

            if not isinstance(myconfig['general-param']['alpha'], float) \
               and not isinstance(myconfig['general-param']['alpha'], int):
                raise ValueError("The alpha regularization parameter for linear"
                                 " variables has to be a float.")
            if myconfig['general-param']['alpha'] < 0:
                raise ValueError("Regularization parameter must be positive.")
            self.alpha = myconfig['general-param']['alpha']

            if myconfig['general-param']['lbound_x'] == '-inf':
                lower_bd_x = -np.inf
            else:
                if not isinstance(myconfig['regul-param']['lbound_x'], float):
                    raise ValueError("The lower bounds on x has to be a float "
                                     "or -inf.")
                lower_bd_x = myconfig['general-param']['lbound_x']
            if myconfig['general-param']['ubound_x'] == 'inf':
                upper_bd_x = np.inf
            else:
                if not isinstance(myconfig['regul-param']['ubound_x'], float):
                    raise ValueError("The upper bounds on x has to be a float "
                                     "or inf.")
                upper_bd_x = myconfig['general-param']['ubound_x']
            if myconfig['general-param']['lbound_y'] == '-inf':
                lower_bd_y = -np.inf
            else:
                if not isinstance(myconfig['regul-param']['lbound_y'], float):
                    raise ValueError("The lower bounds on y has to be a float "
                                     "or -inf.")
                lower_bd_y = myconfig['general-param']['lbound_y']
            if myconfig['general-param']['ubound_y'] == 'inf':
                upper_bd_y = np.inf
            else:
                if not isinstance(myconfig['regul-param']['ubound_y'], float):
                    raise ValueError("The upper bounds on y has to be a float "
                                     "or inf.")
                upper_bd_y = myconfig['general-param']['ubound_y']

            if lower_bd_x >= upper_bd_x:
                raise ValueError("Upper bound on nonlinear variables must be "
                                 "higher than its lower bound.")
            if lower_bd_y >= upper_bd_y:
                raise ValueError("Upper bound on linear variables must be "
                                 "higher than its lower bound.")

            self.bounds_x = (lower_bd_x, upper_bd_x)
            self.bounds_y = (lower_bd_y, upper_bd_y)

            if not isinstance(myconfig['solver-param']['maxit'], int):
                raise ValueError("Maximum number of iterations of the inner "
                                 "solver has to be an integer.")
            if not isinstance(myconfig['solver-param']['tol'], float) \
               and not isinstance(myconfig['solver-param']['tol'], int):
                raise ValueError("Tolerance for the inner solver has to be a "
                                 "float.")
            if myconfig['solver-param']['tol'] <= 0:
                raise ValueError("Solver tolerance must be strictly positive.")
            self.solver_param = SolverParam(myconfig['solver-param']['tol'],
                                            myconfig['solver-param']['maxit'])
        except KeyError:
            print("The configuration file misses the definition of some "
                  "parameters. "
                  "Default values are set for the unknown parameters.")

    def save_initfile(self, filename):
        """Save the parameters to a file in the format of Linux configuration
        file.

        :param filename: Name of the output file
        :type filename: str

        :meta private:
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
        config.set('general-param', 'alpha', str(self.alpha))

        config.add_section('regul-param')
        config.set('regul-param', 'reg_name', str(self.reg.name))
        config.set('regul-param', 'reg_param', str(self.reg.weight))

        config.add_section('solver-param')
        config.set('solver-param', 'tol', str(self.solver_param.gtol))
        config.set('solver-param', 'maxit', str(self.solver_param.maxit))

        with open(filename, 'w') as f:
            config.write(f)

    def save_yaml(self, filename):
        """Save the parameters to a YAML file.

        :param filename: Name of the output file
        :type filename: str

        :meta private:
        """
        mydata = {'general-param': {
                    'maxit': self.maxit,
                    'gtol': self.gtol_h,
                    'verbose': self.verbose,
                    'lbound_x': self.bounds_x[0],
                    'ubound_x': self.bounds_x[1],
                    'lbound_y': self.bounds_y[0],
                    'ubound_y': self.bounds_y[1],
                    'alpha': self.alpha
                  },
                  'regul-param': {
                      'reg_name': self.reg.name,
                      'reg_param': self.reg.weight
                  },
                  'solver-param': {
                      'tol': self.solver_param.gtol,
                      'maxit': self.solver_param.maxit
                   }
                  }

        with open(filename, 'w') as f:
            yaml.dump(mydata, f, default_flow_style=False)
# ============================================================================ #
