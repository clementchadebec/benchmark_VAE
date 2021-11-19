"""This module is the implementation of the Beta_VAE proposed in
(https://openreview.net/pdf?id=Sy2fzU9gl).
This models tweaks add a new parameter to the VAE loss function balancing the weight of the 
reconstruction term and KL term.


Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    :nosignatures:
"""

from .beta_vae_config import BetaVAEConfig
from .beta_vae_model import BetaVAE

__all__ = ["BetaVAE", "BetaVAEConfig"]
