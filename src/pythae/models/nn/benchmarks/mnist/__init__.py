"""A collection of Neural nets used to perform the benchmark on MNIST"""

from .convnets import *
from .resnets import *

__all__ = [
    "Encoder_Conv_AE_MNIST",
    "Encoder_Conv_VAE_MNIST",
    "Encoder_Conv_SVAE_MNIST",
    "Decoder_Conv_AE_MNIST",
    "Discriminator_Conv_MNIST",
    "Encoder_ResNet_AE_MNIST",
    "Encoder_ResNet_VAE_MNIST",
    "Encoder_ResNet_SVAE_MNIST",
    "Encoder_ResNet_VQVAE_MNIST",
    "Decoder_ResNet_AE_MNIST",
    "Decoder_ResNet_VQVAE_MNIST",
]
