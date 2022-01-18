import torch
import numpy as np
import torch.nn as nn
from typing import List

from pythae.models.nn import (
    BaseEncoder,
    BaseDecoder,
    BaseMetric,
    BaseDiscriminator
)
from ..base.base_utils import ModelOutput


class Encoder_AE_MLP(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU())
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(512, self.latent_dim)

    def forward(self, x, output_layer_levels:List[int]=None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(self.depth >= levels > 0 or levels==-1 for levels in output_layer_levels), (
                f'Cannot output layer deeper than depth ({self.depth}). '\
                f'Got ({output_layer_levels}).'
                )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x.reshape(-1, np.prod(self.input_dim))

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'embedding_layer_{i+1}'] = out
            if i+1 == self.depth:
                output['embedding'] = self.embedding(out)

        return output


class Encoder_VAE_MLP(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU())
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(512, self.latent_dim)
        self.log_var = nn.Linear(512, self.latent_dim)

    def forward(self, x, output_layer_levels:List[int]=None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(self.depth >= levels > 0 or levels==-1 for levels in output_layer_levels), (
                f'Cannot output layer deeper than depth ({self.depth}). '\
                f'Got ({output_layer_levels}).'
                )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x.reshape(-1, np.prod(self.input_dim))

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'embedding_layer_{i+1}'] = out
            if i+1 == self.depth:
                output['embedding'] = self.embedding(out)
                output['log_covariance'] = self.log_var(out)

        return output


class Encoder_LadderVAE_MLP(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.latent_dimensions = args.latent_dimensions

        layers = nn.ModuleList()
        mu_s = nn.ModuleList()
        log_var_s = nn.ModuleList()

        layers.append(
            nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU())
        )
        mu_s.append(
            nn.Linear(512, self.latent_dimensions[0])
        )
        log_var_s.append(
            nn.Linear(512, self.latent_dimensions[0])
        )

        layers.append(
            nn.Sequential(nn.Linear(self.latent_dimensions[0], 64), nn.ReLU())
        )
        mu_s.append(
            nn.Linear(64, self.latent_dim)
        )
        log_var_s.append(
            nn.Linear(64, self.latent_dim)
        )

        self.layers = layers
        self.mu_s = mu_s
        self.log_var_s = log_var_s
        self.depth = len(layers)

    def forward(self, x):
        output = ModelOutput()

        out = x.reshape(x.shape[0], -1)

        for i in range(self.depth):
            out = self.layers[i](out)

            mu = self.mu_s[i](out)
            log_var = self.log_var_s[i](out)

            if i+1 == self.depth:
                output['embedding'] = mu
                output['log_covariance'] = log_var

            else:
                output[f'embedding_layer_{i+1}'] = mu
                output[f'log_covariance_layer_{i+1}'] = log_var

            out = mu

        return output


class Decoder_AE_MLP(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim

        # assert 0, np.prod(args.input_dim)

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(args.latent_dim, 512),
                nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Linear(512, int(np.prod(args.input_dim))),
                nn.Sigmoid(),
            )
        )
       
        self.layers = layers
        self.depth = len(layers)


    def forward(self, z: torch.Tensor, output_layer_levels:List[int]=None):

        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(self.depth >= levels > 0 or levels==-1 for levels in output_layer_levels), (
                f'Cannot output layer deeper than depth ({self.depth}). '\
                f'Got ({output_layer_levels}).'
                )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'reconstruction_layer_{i+1}'] = out
            if i+1 == self.depth:
                output['reconstruction'] = out.reshape((z.shape[0],) + self.input_dim)

        return output


class Decoder_LadderVAE_MLP(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.latent_dimensions = args.latent_dimensions

        # assert 0, np.prod(args.input_dim)

        layers = nn.ModuleList()
        mu_s = nn.ModuleList()
        log_var_s = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(args.latent_dim, 64),
                nn.ReLU()
            )
        )
        mu_s.append(
            nn.Linear(64, self.latent_dimensions[-1])
        )
        log_var_s.append(
            nn.Linear(64, self.latent_dimensions[-1])
        )

        layers.append(
            nn.Sequential(
                nn.Linear(self.latent_dimensions[-1], int(np.prod(args.input_dim))),
                nn.Sigmoid(),
            )
        )
       
        self.layers = layers
        self.mu_s = mu_s
        self.log_var_s = log_var_s
        self.depth = len(layers)


    def forward(
        self,
        z: torch.Tensor,
        mu_encoder: List[torch.Tensor],
        log_var_encoder: List[torch.Tensor]
        ):

        mu_encoder.reverse()
        log_var_encoder.reverse()

        output = ModelOutput()

        out = z

        for i in range(self.depth):
            out = self.layers[i](out)

            if i+1 == self.depth:
                output['reconstruction'] = out.reshape((z.shape[0],) + self.input_dim)

            else:
        
                mu_dec = self.mu_s[i](out)
                log_var_dec = self.log_var_s[i](out)

                mu_enc = mu_encoder[i]
                log_var_enc = log_var_encoder[i]

                mu_new, log_var_new = self._update_mu_log_var(
                    mu_enc, log_var_enc, mu_dec, log_var_dec)

                std = torch.exp(0.5 * log_var_new)

                out, _ = self._sample_gauss(mu_new, std)                    

                output[f'embedding_layer_{i+1}'] = mu_new
                output[f'log_covariance_layer_{i+1}'] = log_var_new
                output[f'z_layer_{i+1}'] = out


        return output

    def _update_mu_log_var(self, mu_p, log_var_p, mu_q, log_var_q):
        
        mu_new = (
            mu_p / log_var_p.exp() + mu_q / log_var_q.exp()) / (
                1 / log_var_p.exp() + 1 / log_var_q.exp()
            )

        log_var_new = (1 / (1 / log_var_p.exp() + 1 / log_var_q.exp())).log()

        return (mu_new, log_var_new)

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps 


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

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(np.prod(args.discriminator_input_dim), 256),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
               nn.Linear(256, 1),
                nn.Sigmoid()
            )
        )
       
        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels:List[int]=None):
        """Forward method
        
        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code 
            under the key `reconstruction`
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(self.depth >= levels > 0 or levels==-1 for levels in output_layer_levels), (
                f'Cannot output layer deeper than depth ({self.depth}). '\
                f'Got ({output_layer_levels}).'
                )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z.reshape(z.shape[0], -1)

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'embedding_layer_{i+1}'] = out

            if i+1 == self.depth:
                output['embedding'] = out

        return output
