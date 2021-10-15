"""Implementation of a Normal prior sampler

Available models:

.. autosummary::
    ~pythae.samplers
    ~pythae.models.AE
    :nosignatures:

- |

"""

from .vamp_sampler import VAMPSampler
from .vamp_sampler_config import VAMPSamplerConfig

__all__ = [
   "VAMPSampler",
   "VAMPSamplerConfig"]