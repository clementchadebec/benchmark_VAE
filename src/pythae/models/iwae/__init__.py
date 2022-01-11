"""This module is the implementation of the Importance Weighted Autoencoder
proposed in (https://arxiv.org/abs/1509.00519v4).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    :nosignatures:

"""

from .iwae_model import IWAE
from .iwae_config import IWAEConfig

__all__ = ["IWAE", "IWAEConfig"]
