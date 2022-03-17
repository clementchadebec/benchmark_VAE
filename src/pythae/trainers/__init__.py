""" Here are implemented the trainers used to train the Autoencoder models
"""

from .base_trainer import BaseTrainer, BaseTrainerConfig
from .adversarial_trainer import AdversarialTrainer, AdversarialTrainerConfig
from .coupled_optimizer_trainer import (
    CoupledOptimizerTrainer,
    CoupledOptimizerTrainerConfig,
)

from .coupled_optimizer_adversarial_trainer import (
    CoupledOptimizerAdversarialTrainer,
    CoupledOptimizerAdversarialTrainerConfig,
)

__all__ = [
    "BaseTrainer",
    "BaseTrainerConfig",
    "CoupledOptimizerTrainer",
    "CoupledOptimizerTrainerConfig",
    "AdversarialTrainer",
    "AdversarialTrainerConfig",
    "CoupledOptimizerAdversarialTrainer",
    "CoupledOptimizerAdversarialTrainerConfig",
]
