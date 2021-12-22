from pydantic.dataclasses import dataclass

from ...models import VAEConfig


@dataclass
class VQVAEConfig(VAEConfig):
    r"""
    Vector Quentized VAE model config config class

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        beta (float): The balancing factor in the loss. Default: 1
    """

    beta: float = 0.25
