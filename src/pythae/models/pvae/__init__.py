"""This module is the implementation of a Vanilla Variational Autoencoder
(https://arxiv.org/abs/1312.6114).

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

from .pvae_config import PoincareVAEConfig
from .pvae_model import PoincareVAE

__all__ = ["PoincareVAE", "PoincareVAEConfig"]
