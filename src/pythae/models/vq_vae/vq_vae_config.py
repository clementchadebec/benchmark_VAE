from pydantic.dataclasses import dataclass

from ...models import AEConfig


@dataclass
class VQVAEConfig(AEConfig):
    r"""
    Vector Quentized VAE model config config class

    Parameters:
        input_dim (int): The input_data dimension
        beta (float): The balancing factor in the loss. Default: 1
        num_embedding (int): The number of embedding points. Default: 512
    """
    beta: float = 0.25
    quantization_loss_factor: float = 1.
    num_embeddings: int = 512