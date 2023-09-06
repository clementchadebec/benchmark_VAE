from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class FactorVAEConfig(VAEConfig):
    r"""
    FactorVAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        gamma (float): The balancing factor before the Total Correlation. Default: 0.5
    """
    gamma: float = 2.0
    uses_default_discriminator: bool = True
