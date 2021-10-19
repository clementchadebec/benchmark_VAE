from pydantic.dataclasses import dataclass

from typing_extensions import Literal

from ...models import AEConfig


@dataclass
class RAE_GP_Config(AEConfig):
    """This is the Regularized autoencoder with L2 normalization on the decoder parameters model 
    configuration instance deriving from:class:`~AEConfig`.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        default_encoder (bool): Whether the encoder default. Default: True.
        default_decoder (bool): Whether the encoder default. Default: True.
        embedding_weight (float): The factor before the L2 embedding regularization term in the 
            loss. Default: 1e-4
        reg_weight (float): The factor before the gradient penalty regularization term.
        """
    embedding_weight: float = 1e-4
    reg_weight: float = 1e-7
