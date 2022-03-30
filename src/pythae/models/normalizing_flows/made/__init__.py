"""
Implementation of the Masked Autoencoder model (MADE) proposed
in (https://arxiv.org/abs/1502.03509).
"""

from .made_config import MADEConfig
from .made_model import MADE

__all__ = ["MADE", "MADEConfig"]
