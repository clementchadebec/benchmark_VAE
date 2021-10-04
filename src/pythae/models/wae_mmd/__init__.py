"""Implementation of a Vanilla Auto Encoder

Available samplers:

.. autosummary::
    ~pythae.samplers
    ~pythae.models.AE
    :nosignatures:

- |

"""

from .wae_mmd_model import WAE_MMD
from .wae_mmd_config import WAE_MMD_Config

__all__ = [
   "WAE",
   "WAE_MMD_Config"
   ]