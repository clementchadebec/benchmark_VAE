from pydantic.dataclasses import dataclass

from ...models import VAEConfig


@dataclass
class BetaVAEConfig(VAEConfig):
    r"""
    Gaussian Mixture VAE model config config class

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        number_components (int): The number of components in the mixture. Default: 10
        gaussian_mixture_dim (int): The dimension in which lives the mixture of Gaussian 
            distribution. Default: 10 
    """
    number_components: int = 10
    gaussian_mixture_dim: int = 10
