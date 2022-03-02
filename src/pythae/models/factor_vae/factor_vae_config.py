from pydantic.dataclasses import dataclass

from ...models import VAEConfig


@dataclass
class FactorVAEConfig(VAEConfig):
    r"""
    FactorVAE model config config class

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        gamma (float): The balancing factor before the Total Correlation. Default: 0.5
    """
    gamma: float = 2.
    uses_default_discriminator: bool = True
    discriminator_input_dim: int = None