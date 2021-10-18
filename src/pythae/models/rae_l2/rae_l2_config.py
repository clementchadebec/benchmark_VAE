from pydantic.dataclasses import dataclass

from typing_extensions import Literal

from ...models import AEConfig


@dataclass
class RAE_L2_Config(AEConfig):
    """This is the Regularized autoencoder with L2 normalization on the decoder parameters model 
    configuration instance deriving from:class:`~AEConfig`.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        default_encoder (bool): Whether the encoder default. Default: True.
        default_decoder (bool): Whether the encoder default. Default: True.
        reg_weight (float): The weight decay to apply.
        """
    reg_weight: float = 1e-5
