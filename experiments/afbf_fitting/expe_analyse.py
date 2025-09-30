# -*- coding: utf-8 -*-
r"""Fitting variogram of an anisotropic fractional Brownian field:
    Part 3. Analysis of experiments.
"""
import numpy as np
from afbf.Simulation.TurningBands import LoadTBField
from param_expe_64_evario_512 import params
from matplotlib import pyplot as plt
from os import path

# Repetory for data
home_dir = "/home/frichard/Recherche/Python/varprox/"
# home_dir = "C:/Users/frede/Nextcloud/Synchro/Recherche/Python/varprox/"


def CompareModels(models, models_name, example, display=True):
    """Model Comparison.
    """
    t = np.linspace(- np.pi / 2, np.pi / 2, 10000)
    models[0].hurst.Evaluate(t)
    ref = models[0].hurst.values.reshape((t.size,))
    if display:
        plt.plot(t, ref, "k-", label="reference")
        plt.title("Example " + example)

    bias = []
    rmse = []
    l1er = []
    for j in range(1, len(models)):
        models[j].hurst.Evaluate(t)
        val = models[j].hurst.values.reshape((t.size,))
        bias.append(np.mean(ref - val))
        rmse.append(np.sqrt(np.mean(np.power(ref - val, 2))))
        l1er.append(np.mean(np.abs(ref - val)))
        if display:
            if models_name[j] == "vanilla":
                style = "m:"
            elif models_name[j] == "varproj":
                style = "g--"
            elif models_name[j] == "varprox":
                style = "r-."
            plt.plot(t, val, style, label=models_name[j])
            plt.ylim(0, 1)
            plt.legend()

    if display:
        plt.show()

    return bias, rmse, l1er


# Experience parameters.
param = params()
reg_weight = str(param.reg_param)

nexpe_vanilla = nexpe_varproj = nexpe_varprox = 0
Bias_vanilla = Bias_varproj = Bias_varprox = 0
RMSE_vanilla = RMSE_varproj = RMSE_varprox = 0
L1er_vanilla = L1er_varproj = L1er_varprox = 0

for expe in range(param.Nbexpe):
    caseid = str(expe + 100)
    caseid = caseid[1:]
    file_simu = home_dir + param.data_in + caseid
    file_res = home_dir + param.data_out + caseid

    models = [LoadTBField(file_simu)]
    models_name = ["reference"]

    if path.exists(file_res + "-varproj-noreg-hurst.pickle"):
        models.append(LoadTBField(file_res + "-varproj-noreg"))
        models_name.append("vanilla")

    if path.exists(file_res + "-varproj-hurst.pickle"):
        models.append(LoadTBField(file_res + "-varproj"))
        models_name.append("varproj")

    if path.exists(file_res + "-varprox-" + reg_weight + "-hurst"  + ".pickle"):
        models.append(LoadTBField(file_res + "-varprox" + "-" + reg_weight))
        models_name.append("varprox")

    if len(models_name) > 1:
        bias, rmse, l1er = CompareModels(models, models_name, caseid)
        for j in range(1, len(models_name)):
            j0 = j - 1
            if models_name[j] == "vanilla":
                Bias_vanilla += bias[j0]
                RMSE_vanilla += rmse[j0]
                L1er_vanilla += l1er[j0]
                nexpe_vanilla += 1
            elif models_name[j] == "varprox":
                Bias_varprox += bias[j0]
                RMSE_varprox += rmse[j0]
                L1er_varprox += l1er[j0]
                nexpe_varprox += 1
            elif models_name[j] == "varproj":
                Bias_varproj += bias[j0]
                RMSE_varproj += rmse[j0]
                L1er_varproj += l1er[j0]
                nexpe_varproj += 1


Names = []
Nexpe = []
Bias = []
STD = []
RMSE = []
L1er = []
if nexpe_vanilla > 0:
    Names.append("Vanilla")
    Nexpe.append(nexpe_vanilla)
    bias = Bias_vanilla / nexpe_vanilla
    rmse = RMSE_vanilla / nexpe_vanilla
    std = np.sqrt((rmse)**2 - (bias)**2)
    Bias.append(bias * 100)
    STD.append(std * 100)
    RMSE.append(rmse * 100)
    L1er.append(L1er_vanilla / nexpe_vanilla * 100)

if nexpe_varproj > 0:
    Names.append("Varproj")
    Nexpe.append(nexpe_varproj)
    bias = Bias_varproj / nexpe_varproj
    rmse = RMSE_varproj / nexpe_varproj
    std = np.sqrt((rmse)**2 - (bias)**2)
    Bias.append(bias * 100)
    STD.append(std * 100)
    RMSE.append(rmse * 100)
    L1er.append(L1er_varproj / nexpe_varproj * 100)

if nexpe_varprox > 0:
    Names.append("Varprox")
    Nexpe.append(nexpe_varprox)
    bias = Bias_varprox / nexpe_varprox
    rmse = RMSE_varprox / nexpe_varprox
    std = np.sqrt((rmse)**2 - (bias)**2)
    Bias.append(bias * 100)
    STD.append(std * 100)
    RMSE.append(rmse * 100)
    L1er.append(L1er_varprox / nexpe_varprox * 100)

head = '{:s}: Nexpe =  {:4d}, Bias = {:4.2f}, std = {:4.2f}, RMSE = {:4.2f}, L1 err= {:4.2f}'
for j in range(len(Names)):
    print(head.format(Names[j], Nexpe[j], Bias[j], STD[j], RMSE[j], L1er[j]))
