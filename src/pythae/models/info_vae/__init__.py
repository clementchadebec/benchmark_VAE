"""Implementation of a Info Variational Auto Encoder

Available samplers:

.. autosummary::
    ~pythae.samplers
    ~pythae.models.AE
    :nosignatures:

- |

"""

from .info_vae_model import INFOVAE_MMD
from .info_vae_config import INFOVAE_MMD_Config

__all__ = ["VAE", "VAEConfig"]
