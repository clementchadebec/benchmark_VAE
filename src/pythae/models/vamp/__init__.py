"""This module is the implementation of a Variational Mixture of Posterior prior 
Variational Auto Encoder proposed in (https://arxiv.org/pdf/1705.07120.pdf).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    ~pythae.samplers.VAMPSampler
    :nosignatures:
"""

from .vamp_model import VAMP
from .vamp_config import VAMPConfig

__all__ = ["VAMP", "VAMPConfig"]
