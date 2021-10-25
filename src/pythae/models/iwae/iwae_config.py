from pydantic.dataclasses import dataclass

from ...models import VAEConfig


@dataclass
class IWAEConfig(VAEConfig):
    """This is the Iportance Weighted autoencoder model 
    configuration instance deriving from:class:`VAEConfig`.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        default_encoder (bool): Whether the encoder default. Default: True.
        default_decoder (bool): Whether the encoder default. Default: True.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
         beta (float): The balancing factor between reconstruction and KL. Default: 1
        number_samples (int): Number of samples to use on the Monte-Carlo estimation
    """
    beta: float = 1
    number_samples: int = 10
