"""Sampler fitting a Masked Autoregressive Flow (:class:`pythae.models.normalizing_flows.MAF`) 
in the Autoencoder's latent space.

Available models:
------------------

.. autosummary::
    ~pythae.models.AE
    ~pythae.models.VAE
    ~pythae.models.BetaVAE
    ~pythae.models.VAE
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

from .maf_sampler_config import MAFSamplerConfig
from .maf_sampler import MAFSampler

__all__ = ["MAFSampler", "MAFSamplerConfig"]
