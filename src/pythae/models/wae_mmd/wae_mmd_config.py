from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ...models import AEConfig


@dataclass
class WAE_MMD_Config(AEConfig):
    """Wasserstein autoencoder model config class.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        kernel_choice (str): The kernel to choose. Available options are ['rbf', 'imq'] i.e.
            radial basis functions or inverse multiquadratic kernel. Default: 'imq'.
        reg_weight (float): The weight to apply between reconstruction and Maximum Mean
            Discrepancy. Default: 3e-2
        kernel_bandwidth (float): The kernel bandwidth. Default: 1
    """

    kernel_choice: Literal["rbf", "imq"] = "imq"
    reg_weight: float = 3e-2
    kernel_bandwidth: float = 1.0
