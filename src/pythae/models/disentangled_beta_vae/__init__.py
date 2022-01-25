"""This module is the implementation of the Beta_VAE proposed in
(https://openreview.net/pdf?id=Sy2fzU9gl).
This models adds a new parameter to the VAE loss function balancing the weight of the 
reconstruction term and KL term.


Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    :nosignatures:
"""

from .disentangled_beta_vae_config import DisentangledBetaVAEConfig
from .disentangled_beta_vae_model import DisentangledBetaVAE

__all__ = ["DisentangledBetaVAE", "DisentangledBetaVAEConfig"]
