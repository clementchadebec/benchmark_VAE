"""This module is the implementation of the Gaussian Mixture VAE proposed in
(https://arxiv.org/abs/1611.02648).


Available samplers
-------------------

.. autosummary::

    :nosignatures:
"""

from .gm_vae_config import GMVAEConfig
from .gm_vae_model import GMVAE

__all__ = ["GMVAE", "GMVAEConfig"]
