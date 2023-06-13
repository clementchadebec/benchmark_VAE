from pydantic.dataclasses import dataclass

from ..base.base_config import BaseAEConfig


@dataclass
class AEConfig(BaseAEConfig):
    """This is the autoencoder model configuration instance deriving from
    :class:`~BaseAEConfig`.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        default_encoder (bool): Whether the encoder default. Default: True.
        default_decoder (bool): Whether the encoder default. Default: True.
    """
