# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
r"""
Fit an AFBF model to an image to sample realistic textures.
"""
import numpy as np
from afbf import sdata, coordinates, perfunction, tbfield
from matplotlib import pyplot as plt
from varprox.models.model_afbf import FitVariogram
from numpy.fft import fft2, fftshift
from numpy.random import default_rng

rng = default_rng()

# Original image.
imrep = '../data/'
# imname = 'Mammo/Patch01.png'
imname = 'Film/Patch011.tif'
# imname = 'Misc/mer.tif'

# Parameters to set the lags where to compute the semi-variogram.
N = 40  # size of the grid for the definition of the semi-variogram.
step = 2  # step for grid definition
scalemin = 0  # minimal scale for grid definition of the semi-variogram.

# Parameters for stationary textures.
stationary = None  # True / False if the texture is stationary or not,
# None if its stationarity is to be determined.
sill = 7  # scale at which is the sill starts (only used if stationary=True).

# Model parameters.
K = 16  # Number of parameters for the Hurst function.
J = K  # Number of parameters for the topothesy function.
ftype = "step"  # Type of representation for Hurst and topothesy functions.
noise = 1  # 1 if model with additive noise, 0 otherwise.

# Optimization parameters
multigrid = True
maxit = 10000
gtol = 0.001
verbose = 1

# Evaluation parameters.
spatial_error = True

# Import image.
im = sdata()
im.ImportImage(imrep + imname)
im.name = "Original image"
im.Display(1)
# im.values = (im.values - np.mean(im.values)) / np.std(im.values)

# Lags for the computation of the semi-variogram.
lags = coordinates()
lags.DefineSparseSemiBall(N, step)
sc = np.sqrt(np.power(lags.xy[:, 0], 2) + np.power(lags.xy[:, 1], 2))
ind = np.nonzero(sc > scalemin)
lags.xy = lags.xy[ind[0], :]
sc = sc[ind[0]]
lags.N = im.coord.N

# Empirical semi-variogram of the image.
w0 = im.ComputeEmpiricalSemiVariogram(lags).values[:, 0]

if stationary is None:
    # Automatic detection of a possible sill.
    sigma2 = np.power(np.std(im.values, axis=None), 2)  # image variance.
    g0 = np.zeros(w0.shape)
    g0[:] = 1 - w0[:] / sigma2  # normalize auto-covariance of the field.
    ind = np.nonzero(np.abs(g0) > 0.1)
    sill = np.floor(np.max(sc[ind]))
elif stationary is False:
    sill = N

if sill < N:
    print("Stationary texture: sill at scale %d" % (sill))
    stationary = True
    ind = np.nonzero(sc < sill * 0.8)
    lags_r = coordinates()
    xy = lags.xy[ind[0], :]
    lags_r.DefineNonUniformLocations(xy)
    lags_r.N = lags.N
    w0_r = w0[ind[0]]
    sc_r = sc[ind[0]]
    ind = np.nonzero(sc < sill * 1.2)
    xy = lags.xy[ind[0], :]
    lags.DefineNonUniformLocations(xy)
    lags.N = lags_r.N
    w0 = w0[ind[0]]
    sc = sc[ind[0]]
else:
    print("Non-stationary texture.")
    stationary = False
    sc_r = sc
    w0_r = w0
    lags_r = lags
    scalemax = 1


plt.figure(2)
plt.plot(sc, w0, "g+", sc_r, w0_r, "rx")
plt.title("Empirical semi-variogram.")
plt.show()


# Definition of the reference model.
sc_n = int(sc_r.size / 2)
J = min(J, sc_n)
K = min(K, sc_n)
print('Fitting a model with %d + %d parameters' % (J, K))
topo = perfunction('step', J)
hurst = perfunction('step', K)
model = tbfield("Fitted model", topo, hurst)

# Fitting of the semi-variogram.
model, w1 = FitVariogram(
    model, lags_r, w0_r, noise, None,
    multigrid, maxit, gtol, verbose)
delta = model.noise

plt.figure(3)
plt.plot(sc_r, w0_r, "rx", sc_r, w1, "b+")
plt.title("Pre-fitting.")
plt.show()

# Coordinates where to simulate an image.
coord = coordinates()
coord.DefineUniformGrid(im.coord.N)

if stationary:
    print("Seeking an optimal increment step.")
    # Detect the best increment lag.
    topo = np.zeros(model.topo.fparam.shape)
    hurst = np.zeros(model.hurst.fparam.shape)
    topo[:] = model.topo.fparam[:] / 2
    hurst[:] = model.hurst.fparam[:]

    # Compare the empirical semi-variogram
    # and the increment one at each lag.
    scoremin = np.inf
    kmin = np.zeros((2,))
    wmin = np.zeros(w0.shape)

    for j in range(lags.xy.shape[0]):
        k = lags.xy[j, :]
        model.topo.fparam[:] = topo[:]
        model.hurst.fparam[:] = hurst[:]
        aux, v = FitVariogram(
            model, lags, w0, noise, k,
            False, 0, 0.1, 0)

        score = np.sum(np.power(v - w0, 2))
        if verbose == 1:
            print("increment %d %d: score = %e" % (k[0], k[1], score))
        if score < scoremin:
            scoremin = score
            kmin[:] = k[:]
            wmin[:] = v[:]

    print("Optimal increment: lag = (%d, %d), angle = %e, score = %e"
          % (k[0], k[1], np.arctan(k[1] / k[0]), score))
    k = kmin
    model.topo.fparam[:] = topo[:]
    model.hurst.fparam[:] = hurst[:]
    w2 = wmin
    model, w3 = FitVariogram(model, lags, w0, noise, k,
                             True, maxit, gtol, verbose)
    w3 = w2
    delta = model.noise

    simu = model.Simulate(coord)
    simu = simu.ComputeIncrements(k[0], k[1])
    simu.name = "Simulation"
else:
    simu = model.Simulate(coord)
    w2 = w3 = w1

model.DisplayParameters(3)

simu.name = "Simulation"
if noise == 1:
    simu.values = simu.values +\
        np.sqrt(delta) * rng.standard_normal(simu.values.shape)
simu.Display(5)


# Error analysis on variograms
# Variogram of the estimated model

# model.ComputeApproximateSemiVariogram(lags)
# w4 = model.svario.values[:, 0]
# if noise == 1:
#     w4 = w4 + delta

# Empirical variogram of the simulation.
w11 = simu.ComputeEmpiricalSemiVariogram(lags).values[:, 0]

err = np.power(w11 - w0, 2)
merr = np.sqrt(np.mean(err))
nw0 = np.sqrt(np.mean(np.power(w0, 2)))
print("RMSE=%e (%e percent)" % (merr, merr / nw0 * 100))
err = np.sqrt(err) / nw0 * 100
ind = np.argsort(err)[::-1]
errp = np.concatenate(
    (lags.xy[ind, :], np.expand_dims(sc[ind], 1),
     np.expand_dims(err[ind], 1)),
    axis=1
)
print(errp[0:5, :])

x0 = im.values.reshape(im.M)[0: simu.M[0], 0: simu.M[1]]
x1 = simu.values.reshape(simu.M)

x0f = fft2(x0)
x1f = fft2(x1)

x0p = np.angle(x0f)
x1p = np.angle(x1f)

x0a = np.absolute(x0f)
x1a = np.absolute(x1f)

plt.plot(sc, w0, "r+", label="Semi-variogram of the original image")
# plt.plot(sc, w11, "bo", label="Semi-variogram of the simulated image")
plt.plot(sc, w3, "gx", label="Semi-variogram of the simulation field")
plt.legend(loc="upper left")
plt.xlabel("lag module")
plt.show()

plt.figure(7)
xmin = min(np.min(w0), np.min(w1))
xmax = max(np.max(w1), np.max(w0))
plt.plot(w0, w11, "rx", np.array([xmin, xmax]), np.array([xmin, xmax]), "g--")
plt.xlabel("Semi-variogram of the original image")
plt.ylabel("Semi-variogram of the simulated image")
plt.axis([xmin, xmax, xmin, xmax])

if spatial_error:
    lags2 = coordinates()
    lags2.DefineUniformGrid(N, step, signed=True)
    lags2.N = lags.N

    evario2 = im.ComputeEmpiricalSemiVariogram(lags2)
    model.ComputeApproximateSemiVariogram(lags2)
    evario2.Display(8)
    model.svario.Display(9)

    evario2.values = model.svario.values - evario2.values
    evario2.name = "Variogram difference"
    evario2.Display(10)


# Comparison of original and simulated images.

x0 = im.values.reshape(im.M)[0: simu.M[0], 0: simu.M[1]]
x1 = (x0 - np.mean(x0)) / np.std(x0)
x1 = simu.values.reshape(simu.M)
x1 = (x1 - np.mean(x1)) / np.std(x1)
# Contrast fitting
if contrast_fitting:
    x1 = _match_cumulative_cdf(x1, x0)

x0f = fft2(x0)
x1f = fft2(x1)

x0p = np.angle(x0f)
x1p = np.angle(x1f)

x0a = np.absolute(x0f)
x1a = np.absolute(x1f)

lx0a = np.log(x0a)
lx1a = np.log(x1a)

lxal = 0
lxau = np.max([np.max(lx0a), np.max(lx1a)])


f, axarr = plt.subplots(2, 3, figsize=(40, 20))

axarr[0, 0].imshow(x0, cmap="gray")
axarr[0, 0].set_title("original image: $x_0$", fontsize=50)
axarr[0, 0].axis("off")

axarr[0, 1].imshow(fftshift(x0p), cmap="gray")
axarr[0, 1].set_title("$x_0$: fft phase", fontsize=50)
axarr[0, 1].axis("off")

axarr[0, 2].imshow(fftshift(lx0a), cmap="gray", vmin=lxal, vmax=lxau)
axarr[0, 2].set_title("$x_0$: fft amplitude", fontsize=50)
axarr[0, 2].axis("off")


axarr[1, 0].imshow(x1, cmap="gray")
axarr[1, 0].set_title("simulated image: $x_1$", fontsize=50)
axarr[1, 0].axis("off")

axarr[1, 1].imshow(fftshift(x1p), cmap="gray")
axarr[1, 1].set_title("$x_1$: fft phase", fontsize=50)
axarr[1, 1].axis("off")

axarr[1, 2].imshow(fftshift(lx1a), cmap="gray", vmin=lxal, vmax=lxau)
axarr[1, 2].set_title("$x_1$: fft amplitude", fontsize=50)
axarr[1, 2].axis("off")

plt.show()
