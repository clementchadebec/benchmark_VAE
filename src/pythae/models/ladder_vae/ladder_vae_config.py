from pydantic.dataclasses import dataclass
from typing import List
from ...models import VAEConfig


@dataclass
class LadderVAEConfig(VAEConfig):
    """LadderVAE model config class.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        beta (float): The balancing factor between reconstruction and KL. Default: 1
        ladder_levels (List[int]): The level of the layers to be used in the encoder. Default: [1]
        ladder_latent_dims (List[int]): The latent dimensions to be used in the ladder levels. 
            Defaults: [16]. Note the number of latent dims must match the number of levels in the 
            ladder.
    """
    beta: float = 1
    ladder_levels: List[int] = [1]
    ladder_latent_dims: List[int] = [16]
