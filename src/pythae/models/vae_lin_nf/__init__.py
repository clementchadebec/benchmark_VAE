"""This module is the implementation of a Variational Auto Encoder with Normalizing Flow
(https://arxiv.org/pdf/1505.05770.pdf).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .vae_lin_nf_config import VAE_LinNF_Config
from .vae_lin_nf_model import VAE_LinNF

__all__ = ["VAE_LinNF", "VAE_LinNF_Config"]
