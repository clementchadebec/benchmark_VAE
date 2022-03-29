from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..vae import VAEConfig


@dataclass
class Adversarial_AE_Config(VAEConfig):
    """Adversarial AE model config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        adversarial_loss_scale (float): Parameter scaling the adversarial loss
    """

    adversarial_loss_scale: float = 0.5
    uses_default_discriminator: bool = True
    discriminator_input_dim: int = None
