"""This module implements the trainer to be used when using adversarial models. It uses two distinct
optimizers, one for the encoder, one for the decoder of the AE and one for the discriminator. 
It is suitable for GAN based models models.

Available models:
------------------

.. autosummary::
    ~pythae.models.Adversaral_AE
    :nosignatures:
"""

from .coupled_optimizer_adversarial_trainer import CoupledOptimizerAdversarialTrainer
from .coupled_optimizer_adversarial_trainer_config import CoupledOptimizerAdversarialTrainerConfig

__all__ = ["CoupledOptimizerAdversarialTrainer", "CoupledOptimizerAdversarialTrainerConfig"]
