import typing
from typing import List

import numpy as np
import torch
import torch.nn as nn

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import *


class Layer(nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(self, x):
        return x


class Encoder_AE_Conv(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    self.n_channels, out_channels=32, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(512, self.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}) "
                f"indice. Got ({output_layer_levels})"
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
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))

        return output


class Encoder_VAE_Conv(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    self.n_channels, out_channels=32, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(512, self.latent_dim)
        self.log_var = nn.Linear(512, self.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}) "
                f"indice. Got ({output_layer_levels})"
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
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

        return output


class Encoder_SVAE_Conv(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    self.n_channels, out_channels=32, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(32, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(512, self.latent_dim)
        self.log_concentration = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}) "
                f"indice. Got ({output_layer_levels})"
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
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_concentration"] = self.log_concentration(
                    out.reshape(x.shape[0], -1)
                )

        return output


class Decoder_AE_Conv(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(self.latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    32, out_channels=32, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    32,
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    32,
                    out_channels=self.n_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(self.n_channels),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}) "
                f"indice. Got ({output_layer_levels})"
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

            if i == 0:
                out = out.reshape(z.shape[0], 32, 4, 4)

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output


class Metric_Custom(BaseMetric):
    def __init__(self):
        BaseMetric.__init__(self)


class Encoder_AE_MLP_Custom(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        if args.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of ModelConfig instance must be set to 'data_shape' where"
                "the shape of the data is [mini_batch x data_shape]. Unable to build encoder"
                "automatically"
            )

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(
            nn.Linear(np.prod(args.input_dim), 10), nn.ReLU(), Layer()
        )
        self.mu = nn.Linear(10, self.latent_dim)

    def forward(self, x):
        out = self.layers(x.reshape(-1, int(np.prod(self.input_dim))))

        output = ModelOutput(embedding=self.mu(out))

        return output


class Encoder_VAE_MLP_Custom(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        if args.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of ModelConfig instance must be set to 'data_shape' where"
                "the shape of the data is [mini_batch x data_shape]. Unable to build encoder"
                "automatically"
            )

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(
            nn.Linear(np.prod(args.input_dim), 10), nn.ReLU(), Layer()
        )
        self.mu = nn.Linear(10, self.latent_dim)
        self.std = nn.Linear(10, self.latent_dim)

    def forward(self, x):
        out = self.layers(x.reshape(-1, int(np.prod(self.input_dim))))

        output = ModelOutput(embedding=self.mu(out), log_covariance=self.std(out))

        return output


class Decoder_MLP_Custom(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        if args.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of ModelConfig instance must be set to 'data_shape' where"
                "the shape of the data is [mini_batch x data_shape]. Unable to build decoder"
                "automatically"
            )

        self.layers = nn.Sequential(
            nn.Linear(args.latent_dim, 10),
            nn.ReLU(),
            nn.Linear(10, np.prod(args.input_dim)),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.layers(z)
        output = ModelOutput(reconstruction=out)
        return output


class Metric_MLP_Custom(BaseMetric):
    def __init__(self, args: dict):
        BaseMetric.__init__(self)

        if args.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of ModelConfig instance must be set to 'data_shape' where"
                "the shape of the data is [mini_batch x data_shape]. Unable to build metric"
                "automatically"
            )

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(nn.Linear(np.prod(self.input_dim), 10), nn.ReLU())
        self.diag = nn.Linear(10, self.latent_dim)
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.lower = nn.Linear(10, k)

    def forward(self, x):

        h1 = self.layers(x.reshape(-1, int(np.prod(self.input_dim))))
        h21, h22 = self.diag(h1), self.lower(h1)

        L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(x.device)
        indices = torch.tril_indices(
            row=self.latent_dim, col=self.latent_dim, offset=-1
        )

        # get non-diagonal coefficients
        L[:, indices[0], indices[1]] = h22

        # add diagonal coefficients
        L = L + torch.diag_embed(h21.exp())

        output = ModelOutput(L=L)

        return output


class Discriminator_MLP_Custom(BaseDiscriminator):
    def __init__(self, args: dict):
        BaseDiscriminator.__init__(self)
        self.discriminator_input_dim = args.discriminator_input_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(np.prod(args.discriminator_input_dim), 10),
                nn.ReLU(inplace=True),
            )
        )

        layers.append(
            nn.Linear(10, 5),
        )

        layers.append(nn.Sequential(nn.Linear(5, 1), nn.Sigmoid()))

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}) "
                f"indice. Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z.reshape(z.shape[0], -1)

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["embedding"] = out

        return output


class EncoderWrongInputDim(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.fc = nn.Linear(int(np.prod(args.input_dim)) - 1, args.latent_dim)

    def forward(self, x):
        output = ModelOutput(
            embedding=self.fc(x.reshape(-1, int(np.prod(self.input_dim))))
        )
        return output


class DecoderWrongInputDim(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.latent_dim = args.latent_dim
        self.fc = nn.Linear(args.latent_dim - 1, int(np.prod(args.input_dim)))

    def forward(self, z):
        out = self.fc(z.reshape(-1, self.latent_dim))
        output = ModelOutput(reconstruction=out)
        return output


class MetricWrongInputDim(BaseMetric):
    def __init__(self, args):
        BaseMetric.__init__(self)
        self.input_dim = args.input_dim
        self.fc = nn.Linear(int(np.prod(args.input_dim)) - 1, args.latent_dim)

    def forward(self, x):
        L = self.fc(x.reshape(-1, int(np.prod(self.input_dim))))
        output = ModelOutput(L=L)
        return output


class EncoderWrongOutputDim(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.fc = nn.Linear(int(np.prod(args.input_dim)), args.latent_dim - 1)

    def forward(self, x):
        output = ModelOutput(embedding=self.fc(x.reshape(-1, self.input_dim)))
        return output


class DecoderWrongOutputDim(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.latent_dim = args.latent_dim
        self.fc = nn.Linear(args.latent_dim, int(np.prod(args.input_dim)) - 1)

    def forward(self, z):
        out = self.fc(z.reshape(-1, self.latent_dim))
        output = ModelOutput(reconstruction=out)
        return output


class MetricWrongOutputDim(BaseMetric):
    def __init__(self, args):
        BaseMetric.__init__(self)
        self.input_dim = args.input_dim
        self.fc = nn.Linear(int(np.prod(args.input_dim)), args.latent_dim - 1)

    def forward(self, x):
        L = self.fc(x.reshape(-1, int(np.prod(self.input_dim))))
        output = ModelOutput(L=L)
        return output


class EncoderWrongOutput(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.fc = nn.Linear(int(np.prod(args.input_dim)), args.latent_dim)

    def forward(self, x):
        output = ModelOutput(
            embedding=self.fc(x.reshape(-1, int(np.prod(self.input_dim))))
        )
        return output


class DecoderWrongOutput(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.latent_dim = args.latent_dim
        self.fc = nn.Linear(args.latent_dim, int(np.prod(args.input_dim)))

    def forward(self, z):
        out = self.fc(z.reshape(-1, self.latent_dim))
        output = ModelOutput(reconstruction=out)
        return output, output, output


class MetricWrongOutput(BaseMetric):
    def __init__(self, args):
        BaseMetric.__init__(self)
        self.input_dim = args.input_dim
        self.fc = nn.Linear(int(np.prod(args.input_dim)), args.latent_dim)

    def forward(self, x):
        L = self.fc(x.reshape(-1, self.input_dim))
        output = ModelOutput(L=L)
        return output, output


class MetricWrongOutputDimBis(BaseMetric):
    def __init__(self, args):
        BaseMetric.__init__(self)
        self.latent_dim = args.latent_dim
        self.fc = nn.Linear(int(np.prod(args.input_dim)), args.latent_dim)

    def forward(self, x):
        # out = self.fc(x.reshape(-1, self.input_dim))
        return torch.randn(x.shape[0], self.latent_dim, self.latent_dim - 1)


class NetBadInheritance(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        return 0
