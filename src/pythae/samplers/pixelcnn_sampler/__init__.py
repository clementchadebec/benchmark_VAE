"""Sampler fitting a PixelCNN (:class:`~pythae.models.normalizing_flows.MAF`) 
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
