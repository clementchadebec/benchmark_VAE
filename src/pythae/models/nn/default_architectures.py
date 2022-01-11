import torch
import numpy as np
import torch.nn as nn

from pythae.models.nn import (
    BaseEncoder,
    BaseDecoder,
    BaseMetric,
    BaseDiscriminator,
    BaseLayeredDiscriminator
)
from ..base.base_utils import ModelOutput


class Encoder_AE_MLP(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(nn.Linear(np.prod(args.input_dim), 500), nn.ReLU())
        self.mu = nn.Linear(500, self.latent_dim)

    def forward(self, x):
        out = self.layers(x.reshape(-1, np.prod(self.input_dim)))

        output = ModelOutput(embedding=self.mu(out))

        return output


class Encoder_VAE_MLP(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(nn.Linear(np.prod(args.input_dim), 500), nn.ReLU())
        self.mu = nn.Linear(500, self.latent_dim)
        self.std = nn.Linear(500, self.latent_dim)

    def forward(self, x):
        out = self.layers(x.reshape(-1, np.prod(self.input_dim)))

        output = ModelOutput(embedding=self.mu(out), log_covariance=self.std(out))

        return output


class Decoder_AE_MLP(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim

        # assert 0, np.prod(args.input_dim)

        self.layers = nn.Sequential(
            nn.Linear(args.latent_dim, 500),
            nn.ReLU(),
            nn.Linear(500, int(np.prod(args.input_dim))),
            nn.Sigmoid(),
        )

    def forward(self, z):
        reconstruction = self.layers(z).reshape((z.shape[0],) + self.input_dim)
        output = ModelOutput(reconstruction=reconstruction)
        return output


class Metric_MLP(BaseMetric):
    def __init__(self, args: dict):
        BaseMetric.__init__(self)

        if args.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of ModelConfig instance must be set to 'data_shape' where "
                "the shape of the data is [mini_batch x data_shape]. Unable to build metric "
                "automatically"
            )

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(nn.Linear(np.prod(args.input_dim), 400), nn.ReLU())
        self.diag = nn.Linear(400, self.latent_dim)
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.lower = nn.Linear(400, k)

    def forward(self, x):

        h1 = self.layers(x.reshape(-1, np.prod(self.input_dim)))
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

class Discriminator_MLP(BaseDiscriminator):
    def __init__(self, args: dict):
        BaseDiscriminator.__init__(self)

        self.discriminator_input_dim = args.discriminator_input_dim

        self.layers = nn.Sequential(
            nn.Linear(np.prod(args.discriminator_input_dim), 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layers(x.reshape(-1, np.prod(self.discriminator_input_dim)))

        output = ModelOutput(adversarial_cost=out)

        return output

class LayeredDiscriminator_MLP(BaseLayeredDiscriminator):
    def __init__(self, args: dict):
        
        self.discriminator_input_dim = args.discriminator_input_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(np.prod(args.discriminator_input_dim), 512),
                nn.ReLU()
            )
        )

        layers.append(
            nn.Linear(512, 256),
        )

        layers.append(
            nn.Sequential(
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        )

        BaseLayeredDiscriminator.__init__(self, layers=layers)

    def forward(self, x:torch.Tensor, output_layer_level:int=None):

        if output_layer_level is not None:

            assert output_layer_level <= self.depth, (
                f'Cannot output layer deeper ({output_layer_level}) than depth ({self.depth})'
            )

        x = x.reshape(x.shape[0], -1)

        for i in range(self.depth):
            x = self.layers[i](x)

            if i == output_layer_level:
                break
        
        output = ModelOutput(
            adversarial_cost=x
        )
    
        return output
