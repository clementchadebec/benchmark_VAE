"""A collection of Neural nets used to perform the benchmark on CIFAR"""

from .convnets import *
from .resnets import *

__all__ = [
    "Encoder_Conv_AE_CIFAR",
    "Encoder_Conv_VAE_CIFAR",
    "Encoder_Conv_SVAE_CIFAR",
    "Decoder_Conv_AE_CIFAR",
    "Discriminator_Conv_CIFAR",
    "Encoder_ResNet_AE_CIFAR",
    "Encoder_ResNet_VAE_CIFAR",
    "Encoder_ResNet_SVAE_CIFAR",
    "Encoder_ResNet_VQVAE_CIFAR",
    "Decoder_ResNet_AE_CIFAR",
    "Decoder_ResNet_VQVAE_CIFAR",
]
