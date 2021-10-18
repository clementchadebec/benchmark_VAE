"""This modules is the implementation of the Hamiltonian VAE proposed in
(https://proceedings.neurips.cc/paper/2018/file/3202111cf90e7c816a472aaceb72b0df-Paper.pdf).

This module contains:
    - a :class:`~pythae.models.HVAE` instance which is the implementation of the model.
    - a :class:`~pythae.models.HVAEConfig` instance containing the main parameters of the model
"""

from .hvae_config import HVAEConfig
from .hvae_model import HVAE

__all__ = ["HVAE", "HVAEConfig"]
