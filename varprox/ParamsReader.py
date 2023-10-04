from configparser import ConfigParser
from varprox.models.model_afbf import Fit_Param
import numpy as np


class ParamsReader(object):
    def __init__(self, filename):
        self.config = ConfigParser()
        self.config.read(filename)

    def init_expe_param(self):
        Nbexpe = self.config.getint('expe-param', 'Nbexpe')
        Tvario = self.config.getboolean('expe-param', 'Tvario')
        display = self.config.getboolean('expe-param', 'display')
        save = self.config.getboolean('expe-param', 'save')

        return (Nbexpe, Tvario, display, save)

    def init_model_param(self):
        grid_dim = self.config.getint('model-param', 'grid_dim')
        grid_step = self.config.getint('model-param', 'grid_step')
        field_size = self.config.getint('model-param', 'field_size')
        hurst_dim = self.config.getint('model-param', 'hurst_dim')
        topo_dim = self.config.getint('model-param', 'topo_dim')
        noise = self.config.getint('model-param', 'noise')

        return (grid_dim, grid_step, field_size, hurst_dim, topo_dim, noise)

    def get_optim_param(self):
        noise = self.config.getint('model-param', 'noise')
        multigrid = self.config.getboolean('optim-param', 'multigrid')
        maxit = self.config.getint('optim-param', 'maxit')
        gtol = self.config.getfloat('optim-param', 'gtol')
        verbose = self.config.getboolean('optim-param', 'verbose')
        reg_param = self.config.getfloat('optim-param', 'reg_param')
        alpha = self.config.getfloat('optim-param', 'alpha')
        return Fit_Param(noise, None, multigrid, maxit, gtol, verbose,
                         reg_param, alpha, np.Inf)

