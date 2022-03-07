"""This module implements the dual optimizer trainer using two distinct optimizers for the encoder 
and the decoder. It is suitable for all models but must be used in particular to train a 
:class:`~pythae.models.RAE_L2`.

Available models:
------------------

.. autosummary::
    ~pythae.models.AE
    ~pythae.models.VAE
    ~pythae.models.BetaVAE
    ~pythae.models.DisentangledBetaVAE
    ~pythae.models.BetaTCVAE
    ~pythae.models.IWAE
    ~pythae.models.MSSSIM_VAE
    ~pythae.models.INFOVAE_MMD
    ~pythae.models.WAE_MMD
    ~pythae.models.VAMP
    ~pythae.models.SVAE
    ~pythae.models.VQVAE
    ~pythae.models.RAE_GP
    ~pythae.models.RAE_L2
    ~pythae.models.HVAE
    ~pythae.models.RHVAE
    :nosignatures:
"""

from .coupled_optimizer_trainer import CoupledOptimizerTrainer
from .coupled_optimizer_trainer_config import CoupledOptimizerTrainerConfig

__all__ = ["CoupledOptimizerTrainer", "CoupledOptimizerTrainerConfig"]
