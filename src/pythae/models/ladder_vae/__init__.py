"""This module is the implementation of the Ladder Variational Autoencoder
proposed in (https://arxiv.org/abs/1602.02282).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    :nosignatures:

"""

from .ladder_vae_model import LadderVAE
from .ladder_vae_config import LadderVAEConfig

__all__ = ["LadderVAE", "LadderVAEConfig"]
