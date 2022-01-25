from pydantic.dataclasses import dataclass

from ...models import VAEConfig


@dataclass
class DisentangledBetaVAEConfig(VAEConfig):
    r"""
    :math:Disentangled-`\beta`-VAE model config config class

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        beta (float): The balancing factor. Default: 1000
        C (float): The value of the KL divergence term of the ELBO we wish to approach, measured in nats (see https://en.wikipedia.org/wiki/Nat_(unit)). Default: 50.
        nb_epochs_to_convergence (int): The number of epochs during which the KL divergence objective will incrase up until C (should be smaller or equal to nb_epochs). Default: 100
        epoch (int): The current epoch. Default: 0 
    """

    beta: float = 1000.0
    C: float = 50.0
    nb_epochs_to_convergence: int = 100
    epoch: int = 0
