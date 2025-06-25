# -*- coding: utf-8 -*-
r"""
Tools for computing the semi-variogram of an anisotropic fractional
Brownian field and applying the fitting method.
"""
from numpy import diag, concatenate, zeros, ones, power


# def Ffun(H, scales, logscales, noise=1):

#     F = diag(power(scales[0], H))
#     for p in range(1, scales.size):
#         F = concatenate((F, diag(power(scales[p], H))), axis=0)
#     F = 0.5 * F

#     if noise == 1:
#         F = concatenate((ones((F.shape[0], 1)), F), axis=1)

#     return F


# def DFfun(H, scales, logscales, noise=1):

#     N = scales.size * H.size
#     DF = zeros((N, H.size, H.size))

#     cnt = 0
#     for s in range(scales.size):
#         D = logscales[s] * power(scales[s], H)
#         for j in range(H.size):
#             DF[cnt, j, j] = D[j]
#             cnt += 1

#     if noise == 1:
#         DF = concatenate((zeros((DF.shape[0], 1)), DF), axis=1)


def Ffun_v_block(H, scale, noise):

    F = diag(power(scale, H))
    if noise == 1:
        F = concatenate((ones((F.shape[0], 1)), F), axis=1)

    return F


def Ffun_v(H, scales, logscales, noise=1):

    F = Ffun_v_block(H, scales[0], noise)
    for s in range(1, scales.size):
        F0 = Ffun_v_block(H, scales[s], noise)
        F = concatenate((F, F0), axis=0)

    return F


def DFfun_v(H, c, scales, logscales, noise=1):

    M = H.size
    N = scales.size * M
    DF = zeros((N, M + noise))
    cnt = 0
    for s in range(scales.size):
        D = logscales[s] * power(scales[s], H) * c / N
        for m in range(M):
            DF[cnt, noise + m] = D[m]
            cnt += 1

    return DF
