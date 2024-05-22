from dataclasses import dataclass


@dataclass
class params:
    # Number of experiments
    Nbexpe: int = 100
    # Number of parameters for the Hurst and topothesy functions
    hurst_dim: int = 8
    topo_dim: int = 8
    # Image size.
    N: int = 512
    # Enhancement factor for semi-variogram values.
    enhan_factor: float = 10
    # True if the the theoretical semi-variogram is fitted
    Tvario: bool = True
    # Size of the grid for the definition of the semi-variogram
    grid_dim: int = 40
    # Step for grid definition
    grid_step: int = 2
    # Size of field realization
    grid_normalization: int = 100
    # Display results of each experiment
    display = False
    # Save the results
    save = False
    # 1 if model with noise and 0 otherwise
    noise = 1
    # True if the multigrid algorithm is used.
    multigrid = True
    # Directory for data (Simulated fields)
    data_in = "data/afbf-8/"
    # Directory for results (estimated fields)
    data_out = "experiments/afbf_fitting/results-8-evario/"
