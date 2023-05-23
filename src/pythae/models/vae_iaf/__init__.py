"""This module is the implementation of a Variational Autoencoder with Inverse Autoregressive Flow 
to enhance the expressiveness of the posterior distribution. 
(https://arxiv.org/abs/1606.04934).

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

from .vae_iaf_config import VAE_IAF_Config
from .vae_iaf_model import VAE_IAF

__all__ = ["VAE_IAF", "VAE_IAF_Config"]
