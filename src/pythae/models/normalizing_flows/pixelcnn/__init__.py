"""
Implementation of PixelCNN model proposed
in (https://arxiv.org/abs/1601.06759)
"""

from .pixelcnn_config import PixelCNNConfig
from .pixelcnn_model import PixelCNN

__all__ = ["PixelCNN", "PixelCNNConfig"]
