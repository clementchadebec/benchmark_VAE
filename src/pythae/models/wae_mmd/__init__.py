"""This module is the implementation of the Wasserstein Autoencoder proposed in 
(https://arxiv.org/abs/1711.01558).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .wae_mmd_config import WAE_MMD_Config
from .wae_mmd_model import WAE_MMD

__all__ = ["WAE_MMD", "WAE_MMD_Config"]
