from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class IWAEConfig(VAEConfig):
    """IWAE model config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'l1', 'mse']. Default: 'mse'
        number_samples (int): Number of samples to use on the Monte-Carlo estimation. Default: 10
    """

    number_samples: int = 10
