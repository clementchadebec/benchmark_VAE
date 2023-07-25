from pydantic.dataclasses import dataclass

from ..base import BaseSamplerConfig


@dataclass
class RHVAESamplerConfig(BaseSamplerConfig):
    """RHVAESampler config class.

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
