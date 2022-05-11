from dataclasses import field
from typing import List, Union

from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..vae import VAEConfig


@dataclass
class INFOVAE_MMD_Config(VAEConfig):
    """Info-VAE model config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        kernel_choice (str): The kernel to choose. Available options are ['rbf', 'imq'] i.e.
            radial basis functions or inverse multiquadratic kernel. Default: 'imq'.
        alpha (float): The alpha factor balancing the weigth: Default: 0.5
        lbd (float): The lambda factor. Default: 3e-2
        kernel_bandwidth (float): The kernel bandwidth. Default: 1
        scales (list): The scales to apply if using multi-scale imq kernels. If None, use a unique
            imq kernel. Default: [.1, .2, .5, 1., 2., 5, 10.].
    """

    kernel_choice: Literal["rbf", "imq"] = "imq"
    alpha: float = 0
    lbd: float = 1e-3
    kernel_bandwidth: float = 1.0
    scales: Union[List[float], None] = field(
        default_factory=lambda: [0.1, 0.2, 0.5, 1.0, 2.0, 5, 10.0]
    )
