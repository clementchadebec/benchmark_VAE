from typing import Tuple, Union

from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class VAEGANConfig(VAEConfig):
    """Adversarial AE model config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        adversarial_loss_scale (float): Parameter scaling the adversarial loss
        reconstruction_layer (int): The reconstruction layer depth used for reconstruction metric
    """

    adversarial_loss_scale: float = 0.5
    reconstruction_layer: int = -1
    uses_default_discriminator: bool = True
    discriminator_input_dim: Union[Tuple[int, ...], None] = None
    equilibrium: float = 0.68
    margin: float = 0.4
