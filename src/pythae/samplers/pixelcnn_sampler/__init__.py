"""Sampler fitting a :class:`~pythae.models.normalizing_flows.PixelCNN` 
in the VQVAE's latent space.

Available models:
------------------

.. autosummary::
    ~pythae.models.VQVAE
    :nosignatures:
"""

from .pixelcnn_sampler import PixelCNNSampler
from .pixelcnn_sampler_config import PixelCNNSamplerConfig

__all__ = ["PixelCNNSampler", "PixelCNNSamplerConfig"]
