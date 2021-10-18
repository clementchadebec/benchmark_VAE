"""Implementation of a Vanilla Auto Encoder

Available samplers:

.. autosummary::
    ~pythae.samplers
    ~pythae.models.AE
    :nosignatures:

- |

"""

from .ae_model import AE
from .ae_config import AEConfig

__all__ = ["AE", "AEConfig"]
