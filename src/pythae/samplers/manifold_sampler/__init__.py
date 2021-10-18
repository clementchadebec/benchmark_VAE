"""Implementation of a Manifold sampler proposed in https://arxiv.org/pdf/2105.00026.pdf

Available models:

.. autosummary::
    ~pythae.samplers
    ~pythae.models.RHVAE

"""

from .rhvae_sampler import RHVAESampler
from .rhvae_sampler_config import RHVAESamplerConfig

__all__ = ["NormalSampler", "NormalSamplerConfig"]
