# -*- coding: utf-8 -*-
r"""Fitting variogram of an anisotropic fractional Brownian field:
    Part 3. Analysis of experiments.
"""
import numpy as np
from afbf.Simulation.TurningBands import LoadTBField
from param_expe_64_evario import params

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

RMSE_varproj = 0
RMSE_varprox = 0
for expe in range(7, 8):  # param.Nbexpe):
    caseid = str(expe + 100)
    caseid = caseid[1:]
    file_simu = home_dir + param.data_in + caseid
    file_res = home_dir + param.data_out + caseid

    model_ref = LoadTBField(file_simu)
    model_varproj = LoadTBField(file_res + "-varproj")
    model_varprox = LoadTBField(file_res + "-varprox")

    rmse_varproj = CompareModels(model_ref, model_varproj)
    rmse_varprox = CompareModels(model_ref, model_varprox)

    RMSE_varproj += rmse_varproj
    RMSE_varprox += rmse_varprox
    print('expe {:4d}: rmse varproj = {:.6e}, varprox = {:.6e}'
          .format(expe, rmse_varproj, rmse_varprox))


RMSE_varproj = RMSE_varproj / param.Nbexpe
RMSE_varprox = RMSE_varprox / param.Nbexpe

print('Average RMSE varproj = {:.6e}, varprox = {:.6e}'
      .format(RMSE_varproj, RMSE_varprox))
