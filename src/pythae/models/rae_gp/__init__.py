"""This modules is the implementation of the Regularized AE with gradient penalty regularization
 proposed in (https://arxiv.org/pdf/1903.12436.pdf).

This module contains:
    - a :class:`~pythae.models.RAE_GP` instance which is the implementation of the model.
    - a :class:`~pythae.models.RAE_GP_Config` instance containing the main parameters of the model
"""

from .rae_gp_model import RAE_GP
from .rae_gp_config import RAE_GP_Config

__all__ = ["RAE_GP", "RAE_GP_Config"]
