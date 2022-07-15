"""This module is the implementation of the Vector Quantized VAE proposed in 
(https://arxiv.org/abs/1711.00937).

Available samplers
-------------------

Normalizing flows sampler to come.

.. autosummary::
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .sq_vae_config import SQVAEConfig
from .sq_vae_model import SQVAE

__all__ = ["SQVAE", "SQVAEConfig"]
