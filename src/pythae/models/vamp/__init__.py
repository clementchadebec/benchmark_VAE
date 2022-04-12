"""This module is the implementation of a Variational Mixture of Posterior prior 
Variational Auto Encoder proposed in (https://arxiv.org/abs/1705.07120).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    ~pythae.samplers.VAMPSampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .vamp_config import VAMPConfig
from .vamp_model import VAMP

__all__ = ["VAMP", "VAMPConfig"]
