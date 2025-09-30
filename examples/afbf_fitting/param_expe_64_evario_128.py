from dataclasses import dataclass


@dataclass
class params:
    # Number of experiments
    Nbexpe: int = 100
    # Number of parameters for the Hurst and topothesy functions
    hurst_dim: int = 64
    topo_dim: int = 64
    # Image size.
    N: int = 512
    # Crop image
    crop = None
    # Enhancement factor for semi-variogram values.
    enhan_factor: float = 10
    # True if the the theoretical semi-variogram is fitted
    Tvario: bool = False
    # Size of the grid for the definition of the semi-variogram
    grid_dim: int = 20
    # Step for grid definition
    grid_step: int = 2
    # Size of field realization
    grid_normalization: int = 100
    # Display results of each experiment
    display: bool = False
    # Save the results
    save: bool = False
    # 1 if model with noise and 0 otherwise
    noise: int = 1
    # True if the multigrid algorithm is used.
    multigrid: bool = True
    # Directory for data (Simulated fields)
    data_in: str = "data/afbf-64/"
    # Directory for results (estimated fields)
    data_out: str = "experiments/afbf_fitting/results-64-evario_128/"
    # image crop
    crop: int = 128
    # threshold_reg (for multigrid approach with tv)
    threshold_reg = 32
    reg_param = 1e-4
    alpha = 0.005
    order = 2
