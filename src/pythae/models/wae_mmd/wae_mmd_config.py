from dataclasses import field
from typing import List, Union

from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..ae import AEConfig


@dataclass
class WAE_MMD_Config(AEConfig):
    """Wasserstein autoencoder model config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        kernel_choice (str): The kernel to choose. Available options are ['rbf', 'imq'] i.e.
            radial basis functions or inverse multiquadratic kernel. Default: 'imq'.
        reg_weight (float): The weight to apply between reconstruction and Maximum Mean
            Discrepancy. Default: 3e-2
        kernel_bandwidth (float): The kernel bandwidth. Default: 1
        scales (list): The scales to apply if using multi-scale imq kernels. If None, use a unique
            imq kernel. Default: [.1, .2, .5, 1., 2., 5, 10.].
        reconstruction_loss_scale (float): Parameter scaling the reconstruction loss. Default: 1
    """

    kernel_choice: Literal["rbf", "imq"] = "imq"
    reg_weight: float = 3e-2
    kernel_bandwidth: float = 1.0
    scales: Union[List[float], None] = field(
        default_factory=lambda: [0.1, 0.2, 0.5, 1.0, 2.0, 5, 10.0]
    )
    reconstruction_loss_scale: float = 1.0
