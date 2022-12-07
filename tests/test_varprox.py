#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 12:29:48 2022

@author: frichard
"""



# For test purpose.

# tau = np.zeros(tau0.shape)
# tau[:] = tau0[:]
# beta = np.zeros(beta0.shape)
# beta[:] = beta0[:]

# bounds = (0, np.inf)
# tau = np.random.rand(tau.size)
# pb = minimize(beta0, tau, w0, Ffun, DFfun, bounds, bounds, f, lf, T, B)
# tau1 = pb.argmin_h_y(beta0)
# print('argmin tau')
# print('Error:')
# print(np.mean(np.absolute(tau1 - tau0)))
# # print(tau0)
# # print(tau1)

# beta = np.random.rand(beta.size)
# pb = minimize(beta, tau0, w0, Ffun, DFfun, bounds, bounds, f, lf, T, B)
# beta1 = pb.argmin_h_x(beta, gtol=1e-6, maxit=10000)
# print('argmin beta')
# print('Error:')
# print(np.mean(np.absolute(beta1 - beta0)))
# # print(beta0)
# # print(beta1)