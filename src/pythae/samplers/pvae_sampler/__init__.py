"""Implementation of a the sampling scheme from a Wrapped Riemannian or Riemannian Gaussian 
distribution on the Poincaré Disk as proposed in (https://arxiv.org/abs/1901.06033).

Available models:
------------------

.. autosummary::
    ~pythae.models.PoincareVAE
    :nosignatures:
"""

from .pvae_sampler import PoincareDiskSampler
from .pvae_sampler_config import PoincareDiskSamplerConfig

__all__ = ["PoincareDiskSampler", "PoincareDiskSamplerConfig"]
