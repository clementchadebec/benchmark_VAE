"""This modules is the implementation of the Beta_VAE proposed in
().

This module contains:
    - a :class:`~pythae.models.BetaVAE` instance which is the implementation of the model.
    - a :class:`~pythae.models.BetaVAEConfig` instance containing the main parameters of the model
"""

from .beta_vae_config import BetaVAEConfig
from .beta_vae_model import BetaVAE

__all__ = [
  "BetaVAE",
  "BetaVAEConfig"]
