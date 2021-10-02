from pydantic.dataclasses import dataclass

from pyraug.models.base.base_config import BaseModelConfig, BaseSamplerConfig


@dataclass
class RHVAEConfig(BaseModelConfig):
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


@dataclass
class RHVAESamplerConfig(BaseSamplerConfig):
    """HMCSampler config class containing the main parameters of the sampler.

    Parameters:
        num_samples (int): The number of samples to generate. Default: 1
        batch_size (int): The number of samples per batch. Batching is used to speed up
            generation and avoid memory overflows. Default: 50
        mcmc_steps (int): The number of MCMC steps to use in the latent space HMC sampler.
            Default: 100
        n_lf (int): The number of leapfrog to use in the integrator of the HMC sampler.
            Default: 15
        eps_lf (float): The leapfrog stepsize in the integrator of the HMC sampler. Default: 3e-2
        random_start (bool): Initialization of the latent space sampler. If False, the sampler
            starts the Markov chain on the metric centroids. If True , a random start is applied.
            Default: False
    """

    mcmc_steps_nbr: int = 100
    n_lf: int = 15
    eps_lf: float = 0.03
    beta_zero: float = 1.0
