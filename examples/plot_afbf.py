# -*- coding: utf-8 -*-
r"""Fit an AFBF model to mammograms.
"""
import numpy as np
from afbf import sdata, coordinates, perfunction, tbfield
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift
from numpy.random import default_rng
from varprox import Parameters
from varprox.models.model_afbf import FitVariogram
import time

# Size of the grid for the definition of the semi-variogram
grid_dim = 40
# Step for grid definition
grid_step = 2
# Number of parameters for the Hurst and topothesy function
hurst_dim = topo_dim = 64
# 1 if model with noise and 0 otherwise
noise = 1


# ============================ Plotting Functions =========================== #
def plot_comp_vario(handle, img_semivario, fit_semivario, sim_semivario):
    plt.figure(handle)
    plt.plot(sc, fit_semivario, "bo",
             label="Semi-variogram of the estimated field")
    plt.plot(sc, sim_semivario, "gx",
             label="Semi-variogram of the simulation field")
    plt.legend(loc="upper left")
    plt.xlabel("lag module")
    plt.show()


def plot_comp_vario_2(handle, img_semivario, fit_semivario, sim_semivario):
    plt.figure(handle)
    xmin = np.amin([img_semivario, fit_semivario])
    xmax = np.amax([img_semivario, fit_semivario])
    plt.plot(img_semivario, sim_semivario, "rx", np.array([xmin, xmax]),
             np.array([xmin, xmax]), "g--")
    plt.xlabel("Semi-variogram of the original image")
    plt.ylabel("Semi-variogram of the simulated image")
    plt.axis([xmin, xmax, xmin, xmax])


def plot_comp_fft(x0, x1):
    FONTSIZE = 50

    x0f = fft2(x0)
    x1f = fft2(x1)
    x0p = np.angle(x0f)
    x1p = np.angle(x1f)
    x0a = np.absolute(x0f)
    x1a = np.absolute(x1f)
    lx0a = np.log(x0a)
    lx1a = np.log(x1a)

    lxal = 0
    lxau = np.amax([lx0a, lx1a])

    f, axarr = plt.subplots(2, 3, figsize=(40, 20))

    axarr[0, 0].imshow(x0, cmap="gray")
    axarr[0, 0].set_title("original image: $x_0$", fontsize=FONTSIZE)
    axarr[0, 0].axis("off")

    axarr[0, 1].imshow(fftshift(x0p), cmap="gray")
    axarr[0, 1].set_title("$x_0$: fft phase", fontsize=FONTSIZE)
    axarr[0, 1].axis("off")

    axarr[0, 2].imshow(fftshift(lx0a), cmap="gray", vmin=lxal, vmax=lxau)
    axarr[0, 2].set_title("$x_0$: fft amplitude", fontsize=FONTSIZE)
    axarr[0, 2].axis("off")

    axarr[1, 0].imshow(x1, cmap="gray")
    axarr[1, 0].set_title("simulated image: $x_1$", fontsize=FONTSIZE)
    axarr[1, 0].axis("off")

    axarr[1, 1].imshow(fftshift(x1p), cmap="gray")
    axarr[1, 1].set_title("$x_1$: fft phase", fontsize=50)
    axarr[1, 1].axis("off")

    axarr[1, 2].imshow(fftshift(lx1a), cmap="gray", vmin=lxal, vmax=lxau)
    axarr[1, 2].set_title("$x_1$: fft amplitude", fontsize=FONTSIZE)
    axarr[1, 2].axis("off")

    plt.show()

# =========================================================================== #


if __name__ == "__main__":
    rng = default_rng()

    # Original image.
    IMDIR = '../data/'
    IMNAME = 'Mammo/Patch02.png'

    # Name of the configuration file containing the parameters
    param = Parameters()
    param.load('plot_afbf.ini')
    param.multigrid = True
    param.threshold_reg = 16
    param.noise = 1
    param.alpha = 0

    # Parameters to set the lags where to compute the semi-variogram.
    scalemin = 0  # Minimal scale for grid definition of the semi-variogram.

    # Import image.
    im = sdata()
    im.ImportImage(IMDIR + IMNAME)
    im.name = "Original image"

    # Lags for the computation of the semi-variogram.
    lags = coordinates()
    lags.DefineSparseSemiBall(grid_dim, grid_step)
    sc = np.sqrt(np.power(lags.xy[:, 0], 2) + np.power(lags.xy[:, 1], 2))
    ind = np.nonzero(sc > scalemin)
    lags.xy = lags.xy[ind[0], :]
    sc = sc[ind[0]]
    lags.N = grid_dim * 2

    # Empirical semi-variogram of the image.
    w0 = im.ComputeEmpiricalSemiVariogram(lags).values[:, 0]

    # Definition of the reference model.
    print('Fitting a model with %d + %d parameters' % (topo_dim, hurst_dim))
    topo = perfunction('step', topo_dim)
    hurst = perfunction('step', hurst_dim)
    model = tbfield("Fitted model", topo, hurst)

    start_time = time.perf_counter()
    model, w1 = FitVariogram(model, lags, w0, param)
    end_time = time.perf_counter()
    print("CPU Execution time: {} seconds".format(end_time - start_time))

    model.DisplayParameters(1)

    # Sampling from the estimated field.
    # Coordinates where to simulate an image.
    coord = coordinates()
    coord.DefineUniformGrid(im.coord.N)
    coord.N = lags.N
    simu = model.Simulate(coord)
    delta = model.noise
    simu.name = "Simulation"
    if delta > 0:
        simu.values = simu.values +\
            np.sqrt(delta) * rng.standard_normal(simu.values.shape)

    # Empirical variogram of the simulation.
    w11 = simu.ComputeEmpiricalSemiVariogram(lags).values[:, 0]

    # Compute error
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

    # Plot comparison of original and simulated images.
    x0 = im.values.reshape(im.M)[0:simu.M[0], 0:simu.M[1]]
    x1 = simu.values.reshape(simu.M)
    plot_comp_fft(x0, x1)

    # Plot variogram comparison.
    plot_comp_vario(3, w0, w1, w11)
    plot_comp_vario_2(4, w0, w1, w11)
