"""
**Abstract class**

This is the base AuteEncoder architecture module from which all future autoencoder based 
models should inherit.

It contains:

- | a :class:`~pythae.models.base.base_config.BaseAEConfig` instance containing the main model's
   parameters (*e.g.* latent dimension ...)
- | a :class:`~pythae.models.BaseAE` instance which creates a BaseAE model having a basic
   autoencoding architecture
- | The :class:`~pythae.models.base.base_utils.ModelOutput` instance used for neural nets outputs and 
   model outputs of the :class:`forward` method).
"""

from .base_config import BaseAEConfig
from .base_model import BaseAE

__all__ = ["BaseAE", "BaseAEConfig"]
