# -*- coding: utf-8 -*-
r"""Tools for computing the semi-variogram of a fractional Brownian motion.
"""
from numpy import diag, concatenate, zeros, ones, power


def Ffun(H, scales, logscales, noise=1):

    F = diag(power(scales[0], H))
    for p in range(1, scales.size):
        F = concatenate((F, diag(power(scales[p], H))), axis=0)
    F = 0.5 * F

    if noise == 1:
        F = concatenate((ones((F.shape[0], 1)), F), axis=1)

    return F


def DFfun(H, c, scales, logscales, noise=1):

    M = H.size
    N = scales.size * M
    DF = zeros((N, M + noise))
    cnt = 0
    for s in range(scales.size):
        D = logscales[s] * power(scales[s], H) * c
        for m in range(M):
            DF[cnt, noise + m] = D[m]
            cnt += 1

    return DF
