from pydantic.dataclasses import dataclass

from ...models import VAEConfig


@dataclass
class VAMPConfig(VAEConfig):
    """VAMP config class.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        number_components (int): The number of components to use in the VAMP prior. Default: 50
    """

    number_components: int = 50
