"""This is the base AutoEncoder architecture module from which all future autoencoder based 
models should inherit.
It contains:

- | a :class:`~pythae.models.base.base_config.BaseAEConfig` instance containing the main model's
   parameters (*e.g.* latent dimension ...)
- | a :class:`~pythae.models.BaseAE` instance which creates a BaseAE model having a basic
   autoencoding architecture
- | a :class:`~pythae.models.base.base_config.BaseSamplerConfig` instance containing the main
   sampler's parameters used to sample from the latent space of the BaseAE
- a :class:`~pythae.models.base.base_sampler.BaseSampler` instance which creates a BaseSampler.

.. note::
   If you want ot build upon this work try to make any new model inherit from these 4 classes.
"""

from .base_model import BaseAE
from .base_config import BaseAEConfig

__all__ = ["BaseAE", "BaseAEConfig"]
