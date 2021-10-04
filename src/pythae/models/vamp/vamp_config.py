from pydantic.dataclasses import dataclass

from ...models import VAEConfig

@dataclass
class VAMPConfig(VAEConfig):
    """This is the variational autoencoder with Variational Mixture of Posterior (VAMP) model 
    configuration instance deriving from:class:`VAEConfig`.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        default_encoder (bool): Whether the encoder default. Default: True.
        default_decoder (bool): Whether the encoder default. Default: True.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        number_components (int): The number of components to use in the VAMP prior. Default: 50
        """
    number_components: int = 50