"""This modules is the implementation of the Importance Weighted AE 
 proposed in (https://arxiv.org/pdf/1509.00519v4.pdf).

This module contains:
    - a :class:`~pythae.models.IWAE` instance which is the implementation of the model.
    - a :class:`~pythae.models.IWAEConfig` instance containing the main parameters of the model
"""

from .iwae_model import IWAE
from .iwae_config import IWAEConfig

__all__ = ["IWAE", "IWAEConfig"]
