# -*- coding: utf-8 -*-
r"""Fitting variogram of an anisotropic fractional Brownian field:
    Part 3. Analysis of experiments.
"""
import numpy as np
from afbf.Simulation.TurningBands import LoadTBField
from param_expe_8_evario import params
from matplotlib import pyplot as plt

# Repetory for data
home_dir = "/home/frichard/Recherche/Python/varprox/"
# home_dir = "C:/Users/frede/Nextcloud/Synchro/Recherche/Python/varprox/"


def CompareModels(model_ref, model_varproj, model_varprox, display=True):
    """Model Comparison.
    """

    t = np.linspace(- np.pi / 2, np.pi / 2, 10000)
    hurst_ref = model_ref.hurst
    hurst_varproj = model_varproj.hurst
    hurst_varprox = model_varprox.hurst

    hurst_ref.Evaluate(t)
    hurst_varproj.Evaluate(t)
    hurst_varprox.Evaluate(t)
    ref = hurst_ref.values.reshape((t.size,))
    varproj = hurst_varproj.values.reshape((t.size,))
    varprox = hurst_varprox.values.reshape((t.size,))

    bias_varproj = np.mean(ref - varproj)
    bias_varprox = np.mean(ref - varprox)
    rmse_varproj = np.sqrt(np.mean(np.power(ref - varproj, 2)))
    rmse_varprox = np.sqrt(np.mean(np.power(ref - varprox, 2)))
    l1err_varproj = np.mean(np.abs(ref - varproj))
    l1err_varprox = np.mean(np.abs(ref - varprox))

    if display:
        plt.plot(t, ref, "k-", label="reference")
        plt.plot(t, varproj, "g--", label="varproj estimate")
        plt.plot(t, varprox, "r--", label="varprox estimate")
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

    return bias_varproj, bias_varprox,\
        rmse_varproj, rmse_varprox,\
        l1err_varproj, l1err_varprox


# Experience parameters.
param = params()

Bias_varproj = 0
Bias_varprox = 0
RMSE_varproj = 0
RMSE_varprox = 0
L1err_varproj = 0
L1err_varprox = 0
for expe in range(param.Nbexpe):
    caseid = str(expe + 100)
    caseid = caseid[1:]
    file_simu = home_dir + param.data_in + caseid
    file_res = home_dir + param.data_out + caseid

    model_ref = LoadTBField(file_simu)
    model_varproj = LoadTBField(file_res + "-varproj")
    model_varprox = model_varproj  # LoadTBField(file_res + "-varprox")

    bias_varproj, bias_varprox, rmse_varproj, rmse_varprox,\
        l1err_varproj, l1err_varprox =\
        CompareModels(model_ref, model_varproj, model_varprox)

    Bias_varproj += bias_varproj
    Bias_varprox += bias_varprox
    RMSE_varproj += rmse_varproj
    RMSE_varprox += rmse_varprox
    L1err_varproj += l1err_varproj
    L1err_varprox += l1err_varprox

    print('expe {:4d}: rmse varproj = {:.6e}, varprox = {:.6e}'
          .format(expe, rmse_varproj, rmse_varprox))
    print('L1 error varproj = {:.6e}, L1 error = {:.6e}'
          .format(l1err_varproj, l1err_varprox))

Bias_varproj = Bias_varproj / param.Nbexpe
Bias_varprox = Bias_varprox / param.Nbexpe
RMSE_varproj = RMSE_varproj / param.Nbexpe
RMSE_varprox = RMSE_varprox / param.Nbexpe
L1err_varproj = L1err_varproj / param.Nbexpe
L1err_varprox = L1err_varprox / param.Nbexpe

print('Average Bias varproj = {:.6e}, varprox = {:.6e}'
      .format(Bias_varproj, Bias_varprox))
print('Average RMSE varproj = {:.6e}, varprox = {:.6e}'
      .format(RMSE_varproj, RMSE_varprox))
print('Average L1 error varproj = {:.6e}, varprox = {:.6e}'
      .format(L1err_varproj, L1err_varprox))
