import typing

import torch
import torch.nn as nn
import numpy as np

from pythae.models.nn import *
from pythae.models.base.base_utils import ModelOutput


class Encoder_AE_Conv(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.layers = nn.Sequential(
            nn.Conv2d(
                self.n_channels, out_channels=32, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(nn.Linear(512, 400), nn.ReLU())

        self.mu = nn.Linear(400, self.latent_dim)

    def forward(self, x):
        out = self.layers(x)
        out = self.fc1(out.reshape(x.shape[0], -1))

        output = ModelOutput(embedding=self.mu(out))

        return output


class Encoder_VAE_Conv(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.layers = nn.Sequential(
            nn.Conv2d(
                self.n_channels, out_channels=32, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(nn.Linear(512, 400), nn.ReLU())

        self.mu = nn.Linear(400, self.latent_dim)
        self.std = nn.Linear(400, self.latent_dim)

    def forward(self, x):
        out = self.layers(x)
        out = self.fc1(out.reshape(x.shape[0], -1))

        output = ModelOutput(embedding=self.mu(out), log_covariance=self.std(out))

        return output


class Decoder_AE_Conv(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, 400), nn.ReLU(), nn.Linear(400, 512), nn.ReLU()
        )

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
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

    def forward(self, z):
        out = self.fc1(z)
        reconstruction = self.layers(out.reshape(z.shape[0], 32, 4, 4))
        output = ModelOutput(reconstruction=reconstruction)
        return output


class Metric_Custom(BaseMetric):
    def __init__(self):
        BaseMetric.__init__(self)


class Encoder_MLP_Custom(BaseEncoder):
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

        self.layers = nn.Sequential(nn.Linear(np.prod(args.input_dim), 10), nn.ReLU())
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
