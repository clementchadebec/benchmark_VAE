"""Basic sampler sampling from a uniform distribution on the hypersphere 
in the Autoencoder's latent space.

Available models:
------------------

.. autosummary::
    ~pythae.models.SVAE
    :nosignatures:
"""

from .hypersphere_uniform_config import HypersphereUniformSamplerConfig
from .hypersphere_uniform_sampler import HypersphereUniformSampler

__all__ = ["HypersphereUniformSampler", "HypersphereUniformSamplerConfig"]
