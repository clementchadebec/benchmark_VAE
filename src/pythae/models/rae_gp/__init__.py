"""This module is the implementation of the Regularized AE with gradient penalty regularization
proposed in (https://arxiv.org/abs/1903.12436).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .rae_gp_config import RAE_GP_Config
from .rae_gp_model import RAE_GP

__all__ = ["RAE_GP", "RAE_GP_Config"]
