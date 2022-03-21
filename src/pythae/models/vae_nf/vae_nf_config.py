from pydantic.dataclasses import dataclass

from typing_extensions import Literal

from ..vae import VAEConfig

@dataclass
class VAE_NF_Config(VAEConfig):
    """VAE Normalizing Flow config class.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        """
    pass