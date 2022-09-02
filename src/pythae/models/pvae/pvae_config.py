from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..vae import VAEConfig


@dataclass
class PoincareVAEConfig(VAEConfig):
    """Poincaré VAE config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        prior_distribution (str): The distribution to use as prior
            ["wrapped_normal", "riemannian_normal"]. Default: "wrapped_normal"
        posterior_distribution (str): The distribution to use as posterior
            ["wrapped_normal", "riemannian_normal"]. Default: "wrapped_normal"
        curvature (int): The curvature of the manifold. Default: 1
    """

    prior_distribution: Literal[
        "wrapped_normal", "riemannian_normal"
    ] = "wrapped_normal"
    posterior_distribution: Literal[
        "wrapped_normal", "riemannian_normal"
    ] = "wrapped_normal"
    curvature: float = 1
