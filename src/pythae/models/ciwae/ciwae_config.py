from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class CIWAEConfig(VAEConfig):
    """Combination IWAE model config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        number_samples (int): Number of samples to use on the Monte-Carlo estimation
        beta (float): The value of the factor in the convex combination of the VAE and IWAE ELBO.
            Default: 0.5.
    """

    number_samples: int = 10
    beta: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        assert 0 <= self.beta <= 1, f"Beta parameter must be in [0-1]. Got {self.beta}."
