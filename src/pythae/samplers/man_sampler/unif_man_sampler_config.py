from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..base import BaseSamplerConfig


@dataclass
class UnifManVAESamplerConfig(BaseSamplerConfig):
    """Two Stage VAE sampler config class.

    Parameters:
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        second_stage_depth (int): The number of layers in the second stage VAE. Default: 2
        second_layers_dim (int): The size of the fully connected layer to used un the second VAE
            architecture.
    """

    lbd: float = 1
    tau: float = 1e-10
    mcmc_steps: int = 100
    n_lf: int = 10
    eps_lf: float = 0.01
    n_medoids: int = 10000
