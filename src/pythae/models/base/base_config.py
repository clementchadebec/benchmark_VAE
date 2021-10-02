from pydantic.dataclasses import dataclass

from pyraug.config import BaseConfig


@dataclass
class BaseAEConfig(BaseConfig):
    """This is the base configuration instance of the models deriving from
    :class:`~pyraug.config.BaseConfig`.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        default_encoder (bool): Whether the encoder default. Default: True.
        default_decoder (bool): Whether the encoder default. Default: True.
        """

    input_dim: int = None
    latent_dim: int = 10
    uses_default_encoder: bool = True
    uses_default_decoder: bool = True
