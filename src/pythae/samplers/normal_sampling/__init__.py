"""Basic sampler sampling from a N(0, 1) in the Autoencoder's latent space.

Available models:
------------------

.. autosummary::
    ~pythae.models.AE
    ~pythae.models.VAE
    ~pythae.models.BetaVAE
    ~pythae.models.VAE_LinNF
    ~pythae.models.VAE_IAF
    ~pythae.models.DisentangledBetaVAE
    ~pythae.models.FactorVAE
    ~pythae.models.BetaTCVAE
    ~pythae.models.IWAE
    ~pythae.models.MSSSIM_VAE
    ~pythae.models.WAE_MMD
    ~pythae.models.INFOVAE_MMD
    ~pythae.models.VAMP
    ~pythae.models.SVAE
    ~pythae.models.Adversarial_AE
    ~pythae.models.VAEGAN
    ~pythae.models.VQVAE
    ~pythae.models.HVAE
    ~pythae.models.RAE_GP
    ~pythae.models.RAE_L2
    ~pythae.models.RHVAE
    :nosignatures:
"""

from .normal_config import NormalSamplerConfig
from .normal_sampler import NormalSampler

__all__ = ["NormalSampler", "NormalSamplerConfig"]
