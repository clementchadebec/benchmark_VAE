"""
Implementation of the Masked Autoregressive Flow (MAF) proposed
in (https://arxiv.org/abs/1502.03509)
"""

from .pixelcnn_config import PixelCNNConfig
from .pixelcnn_model import PixelCNN

__all__ = ["PixelCNN", "PixelCNNConfig"]
