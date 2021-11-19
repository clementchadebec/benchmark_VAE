"""This module is the implementation of a Vanilla Variational Autoencoder

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    :nosignatures:
"""

from .vae_model import VAE
from .vae_config import VAEConfig

__all__ = ["VAE", "VAEConfig"]
