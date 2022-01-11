""" Here are implemented the trainers used to train the Autoencoder models
"""

from .base_trainer import BaseTrainer, BaseTrainingConfig
from .coupled_optimizer_trainer import CoupledOptimizerTrainer, CoupledOptimizerTrainerConfig
from .adversarial_trainer import AdversarialTrainer, AdversarialTrainerConfig
from .coupled_optimizer_adversarial_trainer import (
    CoupledOptimizerAdversarialTrainer,
    CoupledOptimizerAdversarialTrainerConfig
)

__all__ = [
    "BaseTrainer",
    "BaseTrainingConfig",
    "CoupledOptimizerTrainer",
    "CoupledOptimizerTrainerConfig",
    "AdversarialTrainer",
    "AdversarialTrainerConfig",
    "CoupledOptimizerAdversarialTrainer",
    "CoupledOptimizerAdversarialTrainerConfig"
]
