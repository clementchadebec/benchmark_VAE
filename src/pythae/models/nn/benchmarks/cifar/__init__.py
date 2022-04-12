"""Neural nets used to perform the benchmark on CIFAR"""

from .convnets import *

__all__ = [
    "Encoder_Conv_AE_CIFAR",
    "Encoder_Conv_VAE_CIFAR",
    "Decoder_Conv_AE_CIFAR"
]
