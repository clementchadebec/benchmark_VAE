"""This module is the implementation of the BetaTCVAE proposed in
(https://arxiv.org/abs/1802.04942).


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

from .beta_tc_vae_config import BetaTCVAEConfig
from .beta_tc_vae_model import BetaTCVAE

__all__ = ["BetaTCVAE", "BetaTCVAEConfig"]
