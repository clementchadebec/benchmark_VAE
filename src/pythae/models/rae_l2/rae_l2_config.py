from pydantic.dataclasses import dataclass

from ..ae import AEConfig


@dataclass
class RAE_L2_Config(AEConfig):
    """RAE_L2 config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        embedding_weight (float): The factor before the L2 regularization term in the loss.
            Default: 1e-4
        reg_weight (float): The weight decay to apply.
    """

    embedding_weight: float = 1e-4
    reg_weight: float = 1e-7
