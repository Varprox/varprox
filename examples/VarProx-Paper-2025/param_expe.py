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
    # True if the the theoretical semi-variogram is fitted
    Tvario: bool = False
    # Size of the grid for the definition of the semi-variogram
    grid_dim: int = 20
    # Display results of each experiment
    display: bool = False
    # Save the results
    save: bool = False
    # 1 if model with noise and 0 otherwise
    noise: int = 1
    # True if the multigrid algorithm is used.
    multigrid: bool = True
    # Directory for data (Simulated fields)
    data_in: str = "data/"
    # Directory for results (estimated fields)
    data_out: str = "results/"
    # image crop
    crop: int = 512
    # threshold_reg (for multigrid approach with tv)
    threshold_reg = 32
    reg_param = 1
    alpha = 0.001
    order = 2
