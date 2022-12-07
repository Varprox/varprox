
# -*- coding: utf-8 -*-
r"""
Experiments of fitting variogram of an anisotropic fractional Brownian field.
Reports
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt

expe = ['01', '04', '08', '16', '32', '64']
radial_prec = np.zeros((len(expe),))
error_l1_beta = np.zeros((len(expe),))

for n in range(len(expe)):
    with open("Results/results-" + expe[n] + "-1024.pickle", "rb") as f:
        Z = pickle.load(f)
        Beta0 = Z[0]
        Tau0 = Z[1]
        Beta1 = Z[2]
        Tau1 = Z[3]
        Tvario = Z[4]
        noise = Z[5]
        multigrid = Z[6]
        maxit = Z[7]
        gtol = Z[8]
        N = Z[9]
        step = Z[10]
        M = Z[11]
        K = Z[12]
        J = Z[13]

        radial_prec[n] = (np.pi - np.pi / K) / np.pi * 100
        error_l1_beta[n] = np.mean(np.absolute(Beta0 - Beta1), axis=None)

plt.plot(radial_prec, error_l1_beta * 100, 'ro--')
plt.xlabel('Radial precision (in percent).')
plt.ylabel(r'Estimation error (in percent).')
plt.axis([0, 100, 0, 8])
plt.show()


expe = ['1024', '512', '256', '128']
image_size = np.zeros((len(expe),))
error_l1_beta = np.zeros((len(expe),))

for n in range(len(expe)):
    with open("Results/results-08-" + expe[n] + ".pickle", "rb") as f:
        Z = pickle.load(f)
        Beta0 = Z[0]
        Tau0 = Z[1]
        Beta1 = Z[2]
        Tau1 = Z[3]
        Tvario = Z[4]
        noise = Z[5]
        multigrid = Z[6]
        maxit = Z[7]
        gtol = Z[8]
        N = Z[9]
        step = Z[10]
        M = Z[11]
        K = Z[12]
        J = Z[13]

        image_size[n] = M
        error_l1_beta[n] = np.mean(np.absolute(Beta0 - Beta1), axis=None)

plt.plot(image_size, error_l1_beta * 100, 'ro--')
plt.legend(loc="upper right")
plt.xlabel('Number of image rows and columns.')
plt.ylabel(r'Estimation error (in percent).')
plt.axis([50, 1030, 0, 30])
plt.show()
