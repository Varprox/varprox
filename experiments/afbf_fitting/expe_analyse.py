# -*- coding: utf-8 -*-
r"""Fitting variogram of an anisotropic fractional Brownian field:
    Part 3. Analysis of experiments.
"""
import numpy as np
from afbf.Simulation.TurningBands import LoadTBField
from param_expe_8_evario import params

# Repetory for data
home_dir = "/home/frichard/Recherche/Python/varprox/"
# home_dir = "C:/Users/frede/Nextcloud/Synchro/Recherche/Python/varprox/"


def CompareModels(model_ref, model_est):
    """Model Comparison.
    """

    t = np.linspace(- np.pi / 2, np.pi / 2, 10000)
    hurst_ref = model_ref.hurst
    hurst_est = model_est.hurst

    hurst_ref.Evaluate(t)
    hurst_est.Evaluate(t)

    rmse = np.sqrt(np.mean(np.power(hurst_ref.values - hurst_est.values, 2)))

    return rmse


# Experience parameters.
param = params()

RMSE = 0
for expe in range(param.Nbexpe):
    caseid = str(expe + 100)
    caseid = caseid[1:]
    file_simu = home_dir + param.data_in + caseid
    file_res = home_dir + param.data_out + caseid

    model_ref = LoadTBField(file_simu)
    model_varproj = LoadTBField(file_res + "-varproj")

    rmse = CompareModels(model_ref, model_varproj)
    RMSE += rmse
    print('expe {:4d}: rmse = {:.6e} '.format(expe, rmse))


RMSE = RMSE / param.Nbexpe
print('Average RMSE = {:.6e} '.format(RMSE))
