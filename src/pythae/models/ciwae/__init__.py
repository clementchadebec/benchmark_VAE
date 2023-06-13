"""This module is the implementation of the Combination Importance Weighted Autoencoder
proposed in (https://arxiv.org/abs/1802.04537).

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

from .ciwae_config import CIWAEConfig
from .ciwae_model import CIWAE

__all__ = ["CIWAE", "CIWAEConfig"]
