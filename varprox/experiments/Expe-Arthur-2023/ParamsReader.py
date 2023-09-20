from configparser import ConfigParser
from varprox.models.model_afbf import Fit_Param
import numpy as np


class ParamsReader(object):
    def __init__(self, filename):
        self.config = ConfigParser()
        self.config.read(filename)

    def init_expe_param(self):
        Nbexpe = self.config.getint('vario-param', 'Nbexpe')
        Tvario = self.config.getboolean('vario-param', 'Tvario')
        noise = self.config.getint('vario-param', 'noise')
        display = self.config.getboolean('vario-param', 'display')
        save = self.config.getboolean('vario-param', 'save')
        stepK = self.config.getint('vario-param', 'stepK')
        return (Nbexpe, Tvario, noise, display, save, stepK)

    def init_model_param(self):
        grid_dim = self.config.getint('model-param', 'grid_dim')
        step = self.config.getint('model-param', 'step')
        field_real = self.config.getint('model-param', 'M')
        hurst_dim = self.config.getint('model-param', 'J')
        return (grid_dim, step, field_real, hurst_dim)

    def get_optim_param(self):
        noise = self.config.getint('vario-param', 'noise')
        multigrid = self.config.getboolean('optim-param', 'multigrid')
        maxit = self.config.getint('optim-param', 'maxit')
        gtol = self.config.getfloat('optim-param', 'gtol')
        verbose = self.config.getboolean('optim-param', 'verbose')
        reg_param = self.config.getfloat('optim-param', 'reg_param')
        alpha = self.config.getfloat('optim-param', 'alpha')
        return Fit_Param(noise, None, multigrid, maxit, gtol, verbose,
                         reg_param, alpha, np.Inf)

