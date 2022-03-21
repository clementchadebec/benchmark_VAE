"""This module is the implementation of the Vector Quantized VAE proposed in 
(https://arxiv.org/abs/1711.00937).

Available samplers
-------------------

Normalizing flows sampler to come.

.. autosummary::

"""

from .vq_vae_config import VQVAEConfig
from .vq_vae_model import VQVAE

__all__ = ["VQVAE", "VQVAEConfig"]
