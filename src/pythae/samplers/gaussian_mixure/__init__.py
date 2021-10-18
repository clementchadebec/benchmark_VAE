"""Implementation of a Gaussian mixture sampler

Available models:

.. autosummary::
    ~pythae.samplers
    ~pythae.models.AE
    :nosignatures:

- |

"""

from .gaussian_mixture_sampler import GaussianMixtureSampler
from .gaussian_mixture_config import GaussianMixtureSamplerConfig

__all__ = ["GaussianMixtureSampler", "GaussianMixtureSamplerConfig"]
