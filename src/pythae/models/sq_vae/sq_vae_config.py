from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class SQVAEConfig(VAEConfig):
    r"""
    Vector Quentized VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        num_embedding (int): The number of embedding points. Default: 512
        temperature_init (float): The init temperature parameter of Gumbel softmax
        temperature_decay (float): The decay in the temperature annealing
    """
    num_embeddings: int = 512
    temperature_init: float = 1
    temperature_decay: float = 1e-5

    def __post_init_post_parse__(self):
        assert 0 <= self.temperature_decay <= 1, (
            "The temperature decay must be in [0, 1]. " f"Got {self.temperature_decay}."
        )

