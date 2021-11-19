"""This module is the implementation of a Info Variational Auto Encoder as proposed in 
(https://arxiv.org/pdf/1706.02262.pdf)

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    :nosignatures:

"""

from .info_vae_model import INFOVAE_MMD
from .info_vae_config import INFOVAE_MMD_Config

__all__ = ["INFOVAE_MMD", "INFOVAE_MMD_Config"]
