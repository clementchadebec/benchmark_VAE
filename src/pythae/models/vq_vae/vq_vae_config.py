from pydantic.dataclasses import dataclass

from ...models import VAEConfig


@dataclass
class VQVAEConfig(AEConfig):
    r"""
    Vector Quentized VAE model config config class

    Parameters:
        input_dim (int): The input_data dimension
        beta (float): The balancing factor in the loss. Default: 1
        embedding_dim (int): The dimension of the embeddings
        num_embedding (int): The number of embedding points
    """
    beta: float = 0.25
    embedding_dim: int = 512
    num_embedding: int = 512