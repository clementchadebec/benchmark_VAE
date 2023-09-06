from pydantic.dataclasses import dataclass

from ..ae import AEConfig


@dataclass
class VQVAEConfig(AEConfig):
    r"""
    Vector Quantized VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        commitment_loss_factor (float): The commitment loss factor in the loss. Default: 0.25.
        quantization_loss_factor: The quantization loss factor in the loss. Default: 1.
        num_embedding (int): The number of embedding points. Default: 512
        use_ema (bool): Whether to use the Exponential Movng Average Update (EMA). Default: False.
        decay (float): The decay to apply in the EMA update. Must be in [0, 1]. Default: 0.99.
    """
    commitment_loss_factor: float = 0.25
    quantization_loss_factor: float = 1.0
    num_embeddings: int = 512
    use_ema: bool = False
    decay: float = 0.99

    def __post_init__(self):
        super().__post_init__()
        if self.use_ema:
            assert 0 <= self.decay <= 1, (
                "The decay in the EMA update must be in [0, 1]. " f"Got {self.decay}."
            )
