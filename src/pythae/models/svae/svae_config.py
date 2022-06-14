from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class SVAEConfig(VAEConfig):
    r"""
    :math:`\mathcal{S}`-VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension in which lives the hypersphere. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
    """
