"""A collection of Neural nets used to perform the benchmark on CELEBA"""

from .convnets import *
from .resnets import *

__all__ = [
    "Encoder_Conv_AE_CELEBA",
    "Encoder_Conv_VAE_CELEBA",
    "Encoder_Conv_SVAE_CELEBA",
    "Decoder_Conv_AE_CELEBA",
    "Discriminator_Conv_CELEBA",
    "Encoder_ResNet_AE_CELEBA",
    "Encoder_ResNet_VAE_CELEBA",
    "Encoder_ResNet_SVAE_CELEBA",
    "Encoder_ResNet_VQVAE_CELEBA",
    "Decoder_ResNet_AE_CELEBA",
    "Decoder_ResNet_VQVAE_CELEBA",
]
