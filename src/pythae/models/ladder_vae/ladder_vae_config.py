from dataclasses import field
from pydantic.dataclasses import dataclass
from typing import List, Union
from ...models import VAEConfig


@dataclass
class LadderVAEConfig(VAEConfig):
    r"""
    Ladder VAE model config config class

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        latent_dimensions (List(int)): The list of latent dimensions to consider in the Ladder. 
            Default: [64] i.e. 1 latent dimension in addition to the final one
            (latent_dim). This fits the 
            `~pythae.models.nn.default_architectures.Encoder_LadderVAE_MLP` neural net.
        beta (float): The balancing factor between reconstruction and KL. Default: 1
        warmup_epoch (float): Number of epoch during whoch beta is increased toward 1. 
            Default: 1
    .. 
        note::

        You must ensure that the length of the latent_dimensions arguments fits the depth of your
        encoder. 

    """
    latent_dimensions: List[int] = field(default_factory=lambda: [64, 32])
    beta: float = 1
    warmup_epoch: int = 1
