"""This module is the implementation of the VAE-GAN model
proposed in (https://arxiv.org/abs/1512.09300).

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

from .vae_gan_config import VAEGANConfig
from .vae_gan_model import VAEGAN

__all__ = ["VAEGAN", "VAEGANConfig"]
