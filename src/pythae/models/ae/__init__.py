"""Implementation of a Vanilla Autoencoder model.

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    :nosignatures:

"""

from .ae_model import AE
from .ae_config import AEConfig

__all__ = ["AE", "AEConfig"]
