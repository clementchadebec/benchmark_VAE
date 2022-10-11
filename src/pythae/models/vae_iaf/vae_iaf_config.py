from typing import Union

from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class VAE_IAF_Config(VAEConfig):
    """VAE with Inverse Autoregressive Normalizing Flow config class.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'.
        n_blocks (int): The number of :class:`~pythae.models.normalizing_flows.IAF` blocks
            to consider in the VAE. Default: 2.
        n_hidden_in_made (int): The number of hidden layers in the
            :class:`~pythae.models.normalizing_flows.MADE` models composing the IAF in the VAE.
            Default: 3.
        hidden_size (list): The number of unit in each hidder layerin the
            :class:`~pythae.models.normalizing_flows.MADE` model. The same number of
            units is used across the `n_hidden_in_made` and `n_blocks`. Default: 128.
        context_dim (tuple): The dimension of the context. Default: None.
    """

    n_blocks: int = 2
    n_hidden_in_made: int = 3
    hidden_size: int = 128
    context_dim: Union[int, None] = None
