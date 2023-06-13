"""Sampler fitting an Inverse Autoregressive Flow (:class:`~pythae.models.normalizing_flows.IAF`) 
in the Autoencoder's latent space.

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

from .iaf_sampler import IAFSampler
from .iaf_sampler_config import IAFSamplerConfig

__all__ = ["IAFSampler", "IAFSamplerConfig"]
