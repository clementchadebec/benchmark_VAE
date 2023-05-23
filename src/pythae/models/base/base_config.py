from typing import Tuple, Union

from pydantic.dataclasses import dataclass

from pythae.config import BaseConfig


@dataclass
class BaseAEConfig(BaseConfig):
    """This is the base configuration instance of the models deriving from
    :class:`~pythae.config.BaseConfig`.

    Parameters:
        input_dim (tuple): The input_data dimension (channels X x_dim X y_dim)
        latent_dim (int): The latent space dimension. Default: None.
    """

    input_dim: Union[Tuple[int, ...], None] = None
    latent_dim: int = 10
    uses_default_encoder: bool = True
    uses_default_decoder: bool = True


@dataclass
class EnvironmentConfig(BaseConfig):
    python_version: str = "3.8"
