from pydantic.dataclasses import dataclass

from ..base.base_config import BaseModelConfig

@dataclass
class VAEConfig(BaseModelConfig):
    """This is the variational autoencoder model configuration instance deriving from
    :class:`~pyraug.config.BaseConfig`.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        default_encoder (bool): Whether the encoder default. Default: True.
        default_decoder (bool): Whether the encoder default. Default: True.
        """
    pass