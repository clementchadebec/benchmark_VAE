from pydantic.dataclasses import dataclass
from typing import Optional

from ..ae import AEConfig


@dataclass
class HRQVAEConfig(AEConfig):
    r"""
    Hierarchical Residual Quantization VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        num_embedding (int): The number of embedding points. Default: 512
        num_levels (int): Depth of hierarchy. Default: 8
        kl_weight (float): Weighting of the KL term. Default: 0.0001
        init_scale (float): Magnitude of the embedding initialisation, should roughly match the encoder. Default: 1.0
        init_decay_weight (float): Factor by which the magnitude of each successive levels is multiplied. Default: 1.5
        norm_loss_weight (float): Weighting of the norm loss term. Default: 0.05
        norm_loss_scale (float): Scale for the norm loss. Default: 1.5
        temp_schedule_gamma (float): Decay constant for the Gumbel temperature - will be (epoch/gamma). Default: 33.333
    """

    num_embeddings: int = 512
    num_levels: int = 8
    kl_weight: float = 0.001
    init_scale: float = 1.0
    init_decay_weight: float = 0.5
    norm_loss_weight: Optional[float] = 0.05
    norm_loss_scale: float = 1.5
    temp_schedule_gamma: float = 33.333
    depth_drop_rate: float = 0.05

    def __post_init__(self):
        super().__post_init__()
