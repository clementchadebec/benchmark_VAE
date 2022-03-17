"""This module implements the trainer to be used when using adversarial models. It uses two distinct
optimizers, one for the encoder and decoder of the AE and one for the discriminator. 
It is suitable for adversarial models.

Available models:
------------------

.. autosummary::
    ~pythae.models.Adversarial_AE
    ~pythae.models.FactorVAE
    :nosignatures:
"""

from .adversarial_trainer import AdversarialTrainer
from .adversarial_trainer_config import AdversarialTrainerConfig

__all__ = ["AdversarialTrainer", "AdversarialTrainerConfig"]
