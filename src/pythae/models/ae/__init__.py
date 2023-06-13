"""Implementation of a Vanilla Autoencoder model.

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:

"""

from .ae_config import AEConfig
from .ae_model import AE

__all__ = ["AE", "AEConfig"]
