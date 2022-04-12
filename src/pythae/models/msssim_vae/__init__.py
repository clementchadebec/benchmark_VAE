"""This module is the implementation of the MSSSIM_VAE proposed in
(https://arxiv.org/abs/1511.06409).
This models uses a perceptual similarity metric for reconstruction.


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

from .msssim_vae_config import MSSSIM_VAEConfig
from .msssim_vae_model import MSSSIM_VAE

__all__ = ["MSSSIM_VAE", "MSSSIM_VAEConfig"]
