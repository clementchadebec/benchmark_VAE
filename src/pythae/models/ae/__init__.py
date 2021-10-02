"""Implementation of a Vanilla Auto Encoder

Available samplers:

.. autosummary::
    ~pyraug.samplers
    ~pyraug.models.AE
    :nosignatures:

- |

"""

from .ae_model import AE
from .ae_config import AEConfig

__all__ = [
   "AE",
   "AEConfig"]