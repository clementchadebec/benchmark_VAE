"""This module is the implementation of a Poincaré Disk Variational Autoencoder
(https://arxiv.org/abs/1901.06033).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.PoincareDiskSampler
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
