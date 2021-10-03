from pydantic.dataclasses import dataclass

from ...models import VAEConfig


@dataclass
class RHVAEConfig(VAEConfig):
    r"""Riemannian Hamiltonian Auto Encoder config class

    Parameters:
        latent_dim (int): The latent dimension used for the latent space. Default: 10
        n_lf (int): The number of leapfrog steps to used in the integrator: Default: 3
        eps_lf (int): The leapfrog stepsize. Default: 1e-3
        beta_zero (int): The tempering factor in the Riemannian Hamiltonian Monte Carlo Sampler.
            Default: 0.3
        temperature (float): The metric temperature :math:`T`. Default: 1.5
        regularization (float): The metric regularization factor :math:`\lambda`
        uses_default_metric (bool): Whether it uses a `custom` or `default` metric architecture.
            This is updated automatically.
        """
    input_dim: int = None
    latent_dim: int = 10
    n_lf: int = 3
    eps_lf: float = 0.001
    beta_zero: float = 0.3
    temperature: float = 1.5
    regularization: float = 0.01
    uses_default_metric: bool = True
