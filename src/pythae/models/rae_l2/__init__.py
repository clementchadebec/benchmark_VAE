"""This module is the implementation of the Regularized AE with L2 decoder parameter regularization
 proposed in (https://arxiv.org/pdf/1903.12436.pdf).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    :nosignatures:
"""

from .rae_l2_model import RAE_L2
from .rae_l2_config import RAE_L2_Config

__all__ = ["RAE_L2", "RAE_L2_Config"]
