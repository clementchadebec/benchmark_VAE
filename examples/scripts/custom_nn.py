from typing import List

import numpy as np
import torch
import torch.nn as nn

from pythae.models import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseDiscriminator, BaseEncoder
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder


class Fully_Conv_Encoder_Conv_AE_MNIST(BaseEncoder):
    """
    A proposed fully Convolutional encoder used for VQVAE.
    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 1, 3, 2, padding=1),
                # nn.BatchNorm2d(128),
                # nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(1, 1, 3, 2, padding=1),
                # nn.BatchNorm2d(256),
                # nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(1, 1, 3, 2, padding=1),
                # nn.BatchNorm2d(512),
                # nn.ReLU(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data under
            the key `embedding`"""
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = out.reshape(x.shape[0], -1)

        return output


class Fully_Conv_Decoder_Conv_AE_MNIST(BaseDecoder):
    """
    A proposed fully Convolutional decoder for VQVAE.
    """

    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (1, 28, 28)
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(1, 1, 3, 2, padding=1),
                # nn.BatchNorm2d(512),
                # nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(1, 1, 3, 2, padding=1, output_padding=1),
                # nn.BatchNorm2d(256),
                # nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    1, self.n_channels, 3, 2, padding=1, output_padding=1
                ),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`
        """
        output = ModelOutput()

        max_depth = self.depth

        z = z.reshape(-1, 1, 4, 4)

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output
