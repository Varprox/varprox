from configparser import ConfigParser
from dataclasses import dataclass
import numpy as np


@dataclass
class SolverParam:
    gtol: float = 1e-3
    maxit: int = 1000


@dataclass
class RegParam:
    name: float = None
    weight: int = 0


class Parameters:
    def __init__(self, gtol=1e-4, maxit=1000, verbose=True, reg=RegParam(),
                 bounds_x=(-np.inf, np.inf), bounds_y=(-np.inf, np.inf),
                 solver_param=SolverParam()):
        self.gtol_h = gtol
        self.maxit = maxit
        self.verbose = verbose
        self.reg = reg
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.solver_param = solver_param

    def __repr__(self):
        mystr = "Object Parameters\n"
        mystr += "  gtol         = {:.3E}\n".format(self.gtol)
        mystr += "  maxit        = {:d}\n".format(self.maxit)
        mystr += "  verbose      = {}\n".format(self.verbose)
        mystr += "  reg          = Name: {} | Weight: {:.3E}\n"\
            .format(self.reg.name, self.reg.weight)
        mystr += "  bounds_x     = {}\n".format(self.bounds_x)
        mystr += "  bounds_y     = {}\n".format(self.bounds_y)
        mystr += "  solver param = Maxit: {} | Tol: {:.3E}\n"\
            .format(self.solver_param.maxit, self.solver_param.gtol)
        return mystr

    def load(self, filename):
        parser = ConfigParser()
        parser.read(filename)

        self.gtol = parser.getfloat('general-param', 'gtol')
        self.maxit = parser.getint('general-param', 'maxit')
        self.verbose = parser.getboolean('general-param', 'verbose')
        self.reg = RegParam(parser.get('regul-param', 'reg_name'),
                            parser.getfloat('regul-param', 'reg_param'))

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
        config = ConfigParser()

        config.add_section('general-param')
        config.set('general-param', 'maxit', str(self.maxit))
        config.set('general-param', 'gtol', str(self.gtol))
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

        config.add_section('solver-param')
        config.set('solver-param', 'tol', str(self.solver_param.gtol))
        config.set('solver-param', 'maxit', str(self.solver_param.maxit))

        with open(filename, 'w') as f:
            config.write(f)
