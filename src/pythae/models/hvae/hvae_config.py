from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class HVAEConfig(VAEConfig):
    r"""Hamiltonian Variational Autoencoder config class.

    Parameters:
        latent_dim (int): The latent dimension used for the latent space. Default: 10
        n_lf (int): The number of leapfrog steps to used in the integrator: Default: 3
        eps_lf (int): The leapfrog stepsize. Default: 1e-3
        beta_zero (int): The tempering factor in the Riemannian Hamiltonian Monte Carlo Sampler.
            Default: 0.3
        learn_eps_lf (bool): Whether the leapfrog stepsize should be learned. Default: False
        learn_beta_zero (bool): Whether the temperature betazero should be learned. Default: False.
    """
    n_lf: int = 3
    eps_lf: float = 0.001
    beta_zero: float = 0.3
    learn_eps_lf: bool = False
    learn_beta_zero: bool = False
