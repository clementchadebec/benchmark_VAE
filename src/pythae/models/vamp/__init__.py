"""Implementation of a VAMP prior Variational Auto Encoder

Available samplers:

.. autosummary::
    ~pythae.samplers
    ~pythae.models.AE
    :nosignatures:

- |

"""

from .vamp_model import VAMP
from .vamp_config import VAMPConfig

__all__ = [
   "VAMP",
   "VAMPConfig"]