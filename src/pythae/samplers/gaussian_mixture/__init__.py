"""Implementation of a Gaussian mixture sampler.

Available models:
------------------

.. autosummary::
    ~pythae.models.AE
    ~pythae.models.VAE
    ~pythae.models.BetaVAE
    ~pythae.models.DisentangledBetaVAE
    ~pythae.models.IWAE
    ~pythae.models.WAE_MMD
    ~pythae.models.INFOVAE_MMD
    ~pythae.models.VAMP
    ~pythae.models.Adversarial_AE
    ~pythae.models.VAEGAN
    ~pythae.models.HVAE
    ~pythae.models.RAE_GP
    ~pythae.models.RAE_L2
    ~pythae.models.RHVAE
    :nosignatures:
"""

from .gaussian_mixture_sampler import GaussianMixtureSampler
from .gaussian_mixture_config import GaussianMixtureSamplerConfig

__all__ = ["GaussianMixtureSampler", "GaussianMixtureSamplerConfig"]
