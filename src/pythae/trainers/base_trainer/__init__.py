"""This module implements the base trainer allowing you to train the models implemented in pythae.

Available models:
------------------

.. autosummary::
    ~pythae.models.AE
    ~pythae.models.VAE
    ~pythae.models.BetaVAE
    ~pythae.models.INFOVAE_MMD
    ~pythae.models.RAE_GP
    ~pythae.models.WAE_MMD
    ~pythae.models.VAMP
    ~pythae.models.HVAE
    ~pythae.models.RHVAE
    :nosignatures:
"""

from .base_trainer import BaseTrainer
from .base_training_config import BaseTrainingConfig

__all__ = ["BaseTrainer", "BaseTrainingConfig"]
