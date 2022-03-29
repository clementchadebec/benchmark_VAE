from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class DisentangledBetaVAEConfig(VAEConfig):
    r"""
    Disentangled :math:`\beta`-VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        beta (float): The balancing factor. Default: 10.
        C (float): The value of the KL divergence term of the ELBO we wish to approach, measured in
            nats. Default: 50.
        warmup_epoch (int): The number of epochs during which the KL divergence objective will
            increase from 0 to C (should be smaller or equal to nb_epochs). Default: 100
        epoch (int): The current epoch. Default: 0
    """
    beta: float = 10.0
    C: float = 50.0
    warmup_epoch: int = 25
