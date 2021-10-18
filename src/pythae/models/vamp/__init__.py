"""This modules is the implementation of a VAMP prior Variational Auto Encoder proposed in 
https://arxiv.org/pdf/1705.07120.pdf.

Available samplers:

.. autosummary::
    ~pythae.samplers
    ~pythae.models.AE
    :nosignatures:

- |

"""

from .vamp_model import VAMP
from .vamp_config import VAMPConfig

__all__ = ["VAMP", "VAMPConfig"]
