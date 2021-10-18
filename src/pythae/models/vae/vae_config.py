from pydantic.dataclasses import dataclass

from typing_extensions import Literal

from ..base.base_config import BaseAEConfig


@dataclass
class VAEConfig(BaseAEConfig):
    """This is the variational autoencoder model configuration instance deriving from
    :class:`~pythae.config.BaseConfig`.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        default_encoder (bool): Whether the encoder default. Default: True.
        default_decoder (bool): Whether the encoder default. Default: True.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        """

    reconstruction_loss: Literal["bce", "mse"] = "mse"
