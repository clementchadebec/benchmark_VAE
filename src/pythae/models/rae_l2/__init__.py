"""This modules is the implementation of the Regularized AE with L2 decoder parameter regularization
 proposed in (https://arxiv.org/pdf/1903.12436.pdf).

This module contains:
    - a :class:`~pythae.models.RAE_L2` instance which is the implementation of the model.
    - a :class:`~pythae.models.RAE_L2_Config` instance containing the main parameters of the model
"""

from .rae_l2_model import RAE_L2
from .rae_l2_config import RAE_L2_Config

__all__ = ["RAE_L2", "RAE_L2_Config"]
