"""This module is the implementation of the Vector Qauntized VAE proposed in 
(https://arxiv.org/abs/1711.00937).

Available samplers
-------------------

.. autosummary::

"""

from .vq_vae_model import VQVAE
from .vq_vae_config import VQVAE_Config

__all__ = ["VQVAE", "VQVAEConfig"]
