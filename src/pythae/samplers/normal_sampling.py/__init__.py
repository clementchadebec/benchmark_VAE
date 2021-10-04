"""Implementation of a Normal prior sampler

Available models:

.. autosummary::
    ~pythae.samplers
    ~pythae.models.AE
    :nosignatures:

- |

"""

from .normal_sampler import NormalSampler
from .normal_config import NormalSampler_Config

__all__ = [
   "NormalSampler",
   "NormalSampler_Config"]