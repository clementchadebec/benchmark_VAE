from pydantic.dataclasses import dataclass

from typing_extensions import Literal

from ...models import VAEConfig


@dataclass
class INFOVAE_MMD_Config(VAEConfig):
    """This is the Info Variational Autoencoder model configuration instance deriving from
    :class:`~AEConfig`.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        default_encoder (bool): Whether the encoder default. Default: True.
        default_decoder (bool): Whether the encoder default. Default: True.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        kernel_choice (str): The kernel to choose. Available options are ['rbf', 'imq'] i.e. 
            radial basis functions or inverse multiquadratic kernel. Default: 'imq'.
        alpha (float): The alpha factor balancing the weigth: Default: 0.5
        lbd (float): The lambda factor. Default: 3e-2
        kernel_bandwidth (float): The kernel bandwidth. Default: 1
        """

    kernel_choice: Literal["rbf", "imq"] = "imq"
    alpha: float = 0.5
    lbd: float = 3e-2
    kernel_bandwidth: float = 1.0
