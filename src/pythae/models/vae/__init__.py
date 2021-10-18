"""Implementation of a Vanilla Variational Auto Encoder

Available samplers:

.. autosummary::
    ~pythae.samplers
    ~pythae.models.AE
    :nosignatures:

- |

"""

from .vae_model import VAE
from .vae_config import VAEConfig

__all__ = ["VAE", "VAEConfig"]
