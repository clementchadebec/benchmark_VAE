"""Basic sampler sampling from a N(0, 1) in the Autoencoder's latent space.

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

from .normal_sampler import NormalSampler
from .normal_config import NormalSamplerConfig

__all__ = ["NormalSampler", "NormalSamplerConfig"]
