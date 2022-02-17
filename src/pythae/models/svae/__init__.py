"""This module is the implementation of the Hyperspherical VAE proposed in
(https://arxiv.org/abs/1804.00891).
This models uses a Hyperspherical latent space.


Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.HypersphereUniformSampler
    :nosignatures:
"""

from .svae_config import SVAEConfig
from .svae_model import SVAE

__all__ = ["SVAE", "SVAEConfig"]
