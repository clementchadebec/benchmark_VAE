from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class BetaTCVAEConfig(VAEConfig):
    r"""
    :math:`\beta`-TCVAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        alpha (float): The balancing factor before the Index code Mutual Info. Default: 1
        beta (float): The balancing factor before the Total Correlation. Default: 1
        gamma (float): The balancing factor before the dimension-wise KL. Default: 1
        use_mss (bool): Use Minibatch Stratified Sampling. If False: uses Minibatch Weighted
            Sampling. Default: True
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    use_mss: bool = True
