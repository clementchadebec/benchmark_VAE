"""Proposed neural nets architectures suited for MNIST"""

import torch
import torch.nn as nn
import numpy as np

from typing import List
from ..base_architectures import BaseEncoder, BaseDecoder
from ....models.base.base_utils import ModelOutput
from ....models import BaseAEConfig

from pythae.models.nn import (
    BaseEncoder,
    BaseDecoder,
    BaseDiscriminator
)


class Encoder_AE_MNIST(BaseEncoder):
    """
    A proposed Convolutional encoder Neural net suited for MNIST and Autoencoder-based models.

    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.mnist import Encoder_AE_MNIST
            >>> from pythae.models import AEConfig
            >>> model_config = AEConfig(input_dim=(1, 28, 28), latent_dim=16)
            >>> encoder = Encoder_AE_MNIST(model_config)
            >>> encoder
            ... Encoder_AE_MNIST(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...   )
            ...   (embedding): Linear(in_features=1024, out_features=16, bias=True)
            ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import AE
        >>> model = AE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True

    .. note::

        Please note that this encoder is only suitable for Autoencoder based models since it only
        outputs the embeddings of the input data under the key `embedding`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 1, 28, 28)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 16])


    """
    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(1024, args.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels:List[int]=None):
        """Forward method
        
        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data 
            under the key `embedding`. Optional: The outputs of the layers specified in 
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(self.depth >= levels > 0 or levels==-1 for levels in output_layer_levels), (
                f'Cannot output layer deeper than depth ({self.depth}).'\
                f'Got ({output_layer_levels}).'
                )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'embedding_layer_{i+1}'] = out
            if i+1 == self.depth:
                output['embedding'] = self.embedding(out.reshape(x.shape[0], -1))

        return output


class Encoder_VAE_MNIST(BaseEncoder):
    """
    A Convolutional encoder Neural net suited for MNIST and Variational Autoencoder-based 
    models.


    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.mnist import Encoder_VAE_MNIST
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(1, 28, 28), latent_dim=16)
            >>> encoder = Encoder_VAE_MNIST(model_config)
            >>> encoder
            ... Encoder_VAE_MNIST(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...   )
            ...   (embedding): Linear(in_features=1024, out_features=16, bias=True)
            ...   (log_var): Linear(in_features=1024, out_features=16, bias=True)
            ... )

    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True

    .. note::

        Please note that this encoder is only suitable for Variational Autoencoder based models 
        since it outputs the embeddings and the **log** of the covariance diagonal coefficients 
        of the input data under the key `embedding` and `log_covariance`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 1, 28, 28)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 16])
            >>> out.log_covariance.shape
            ... torch.Size([2, 16])


    """
    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(1024, args.latent_dim)
        self.log_var = nn.Linear(1024, args.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels:List[int]=None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.
        
        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data 
            under the key `embedding` and the **log** of the diagonal coefficient of the covariance 
            matrices under the key `log_covariance`. Optional: The outputs of the layers specified 
            in `output_layer_levels` arguments are available under the keys `embedding_layer_i` 
            where i is the layer's level."""
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(self.depth >= levels > 0 or levels==-1 for levels in output_layer_levels), (
                f'Cannot output layer deeper than depth ({self.depth}).'\
                f'Got ({output_layer_levels})'
                )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'embedding_layer_{i+1}'] = out
        
            if i+1 == self.depth:
                output['embedding'] = self.embedding(out.reshape(x.shape[0], -1))
                output['log_covariance'] = self.log_var(out.reshape(x.shape[0], -1))

        return output


class Encoder_LadderVAE_MNIST(BaseEncoder):
    """
    A Convolutional encoder Neural net suited for MNIST and Variational Autoencoder-based 
    models.


    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.mnist import Encoder_VAE_MNIST
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(1, 28, 28), latent_dim=16)
            >>> encoder = Encoder_VAE_MNIST(model_config)
            >>> encoder
            ... Encoder_VAE_MNIST(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...   )
            ...   (embedding): Linear(in_features=1024, out_features=16, bias=True)
            ...   (log_var): Linear(in_features=1024, out_features=16, bias=True)
            ... )

    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True

    .. note::

        Please note that this encoder is only suitable for Variational Autoencoder based models 
        since it outputs the embeddings and the **log** of the covariance diagonal coefficients 
        of the input data under the key `embedding` and `log_covariance`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 1, 28, 28)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 16])
            >>> out.log_covariance.shape
            ... torch.Size([2, 16])


    """
    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.latent_dimensions = args.latent_dimensions
        self.n_channels = 1

        layers = nn.ModuleList()
        mu_s = nn.ModuleList()
        log_var_s = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        mu_s.append(
            nn.Linear(128 * 14 * 14, self.latent_dimensions[0])
        )
        log_var_s.append(
            nn.Linear(128 * 14 * 14, self.latent_dimensions[0])
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )
        mu_s.append(
            nn.Linear(256 * 7 * 7, self.latent_dimensions[1])
        )
        log_var_s.append(
            nn.Linear(256 * 7 * 7, self.latent_dimensions[1])
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )
        mu_s.append(
            nn.Linear(512 * 3 * 3, self.latent_dimensions[2])
        )
        log_var_s.append(
            nn.Linear(512 * 3 * 3, self.latent_dimensions[2])
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            )
        )
        mu_s.append(
            nn.Linear(1024, self.latent_dim)
        )
        log_var_s.append(
            nn.Linear(1024, self.latent_dim)
        )

        self.layers = layers
        self.mu_s = mu_s
        self.log_var_s = log_var_s
        self.depth = len(layers)

    

    def forward(self, x: torch.Tensor):

        output = ModelOutput()

        max_depth = self.depth

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            mu = self.mu_s[i](out.reshape(x.shape[0], -1))
            log_var = self.log_var_s[i](out.reshape(x.shape[0], -1))

            if i+1 == self.depth:
                output['embedding'] = mu
                output['log_covariance'] = log_var
        
            else:
                output[f'embedding_layer_{i+1}'] = mu
                output[f'log_covariance_layer_{i+1}'] = log_var

        return output


class Decoder_AE_MNIST(BaseDecoder):
    """
    A proposed Convolutional decoder Neural net suited for MNIST and Autoencoder-based 
    models.

    .. code-block::

            >>> from pythae.models.nn.benchmarks.mnist import Decoder_AE_MNIST
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(1, 28, 28), latent_dim=16)
            >>> decoder = Decoder_AE_MNIST(model_config)
            >>> decoder
            ... Decoder_AE_MNIST(
            ...   (layers): ModuleList(
            ...     (0): Linear(in_features=16, out_features=16384, bias=True)
            ...     (1): Sequential(
            ...       (0): ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): ConvTranspose2d(256, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
            ...       (1): Sigmoid()
            ...     )
            ...   )
            ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, decoder=decoder)
        >>> model.decoder == decoder
        ... True

    .. note::

        Please note that this decoder is suitable for **all** models.

        .. code-block::

            >>> import torch
            >>> input = torch.randn(2, 16)
            >>> out = decoder(input)
            >>> out.reconstruction.shape
            ... torch.Size([2, 1, 28, 28])
    """
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Linear(args.latent_dim, 1024 * 4 * 4)
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 3, 2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 3, 2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )
        
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, self.n_channels, 3, 2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels:List[int]=None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.
        
        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code 
            under the key `reconstruction`. Optional: The outputs of the layers specified in 
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i` 
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(self.depth >= levels > 0 or levels==-1 for levels in output_layer_levels), (
                f'Cannot output layer deeper than depth ({self.depth}).'\
                f'Got ({output_layer_levels})'
                )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if i == 0:
                out = out.reshape(z.shape[0], 1024, 4, 4)

            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'reconstruction_layer_{i+1}'] = out

            if i+1 == self.depth:
                output['reconstruction'] = out

        return output


class Decoder_LadderVAE_MNIST(BaseDecoder):
    """
    A proposed Convolutional decoder Neural net suited for MNIST and Autoencoder-based 
    models.

    .. code-block::

            >>> from pythae.models.nn.benchmarks.mnist import Decoder_AE_MNIST
            >>> from pythae.models import VAEConfig
            >>> model_config = VAEConfig(input_dim=(1, 28, 28), latent_dim=16)
            >>> decoder = Decoder_AE_MNIST(model_config)
            >>> decoder
            ... Decoder_AE_MNIST(
            ...   (layers): ModuleList(
            ...     (0): Linear(in_features=16, out_features=16384, bias=True)
            ...     (1): Sequential(
            ...       (0): ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            ...       (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (2): Sequential(
            ...       (0): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
            ...       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...       (2): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): ConvTranspose2d(256, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
            ...       (1): Sigmoid()
            ...     )
            ...   )
            ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, decoder=decoder)
        >>> model.decoder == decoder
        ... True

    .. note::

        Please note that this decoder is suitable for **all** models.

        .. code-block::

            >>> import torch
            >>> input = torch.randn(2, 16)
            >>> out = decoder(input)
            >>> out.reconstruction.shape
            ... torch.Size([2, 1, 28, 28])
    """
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.latent_dimensions = args.latent_dimensions
        self.n_channels = 1

        layers = nn.ModuleList()
        ladder_layers = nn.ModuleList()
        mu_s = nn.ModuleList()
        log_var_s = nn.ModuleList()

        # ladder
        ladder_layers.append(
            nn.Sequential(
                nn.Linear(args.latent_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
        )
        mu_s.append(
            nn.Linear(64, self.latent_dimensions[-1])
        )
        log_var_s.append(
            nn.Linear(64, self.latent_dimensions[-1])
        )

        ladder_layers.append(
            nn.Sequential(
                nn.Linear(self.latent_dimensions[-1], 128),
                nn.BatchNorm1d(128),
                nn.ReLU()
            )
        )
        mu_s.append(
            nn.Linear(128, self.latent_dimensions[-2])
        )
        log_var_s.append(
            nn.Linear(128, self.latent_dimensions[-2])
        )

        ladder_layers.append(
            nn.Sequential(
                nn.Linear(self.latent_dimensions[-2], 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )
        )
        mu_s.append(
            nn.Linear(256, self.latent_dimensions[-3])
        )
        log_var_s.append(
            nn.Linear(256, self.latent_dimensions[-3])
        )


        # decoding
        layers.append(
            nn.Linear(self.latent_dimensions[-3], 1024 * 4 * 4)
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 3, 2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 3, 2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )
        
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, self.n_channels, 3, 2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.ladder_layers = ladder_layers
        self.mu_s = mu_s
        self.log_var_s = log_var_s
        self.depth = len(layers)
        self.ladder_depth = len(ladder_layers)

    def forward(self,
        z: torch.Tensor,
        mu_encoder: List[torch.Tensor]=None,
        log_var_encoder: List[torch.Tensor]=None
        ):

        if mu_encoder is not None:
            mu_encoder.reverse()

        if log_var_encoder is not None:
            log_var_encoder.reverse()

        output = ModelOutput()

        out = z

        # encoding ladder
        for i in range(self.ladder_depth):
           
            out = self.ladder_layers[i](out)

            mu_dec = self.mu_s[i](out)
            log_var_dec = self.log_var_s[i](out)

            if mu_encoder is None or log_var_encoder is None:
                    mu_new, log_var_new = mu_dec, log_var_dec

            else:
                mu_enc = mu_encoder[i]
                log_var_enc = log_var_encoder[i]
                mu_new, log_var_new = self._update_mu_log_var(
                mu_enc.detach(), log_var_enc.detach(), mu_dec, log_var_dec)

            std = torch.exp(0.5 * log_var_new)
               
            out, _ = self._sample_gauss(mu_new, std)


            output[f'embedding_layer_{i+1}'] = mu_dec
            output[f'log_covariance_layer_{i+1}'] = log_var_dec
            output[f'z_layer_{i+1}'] = out

        for i in range(self.depth):
            out = self.layers[i](out)

            if i == 0:
                out = out.reshape(z.shape[0], 1024, 4, 4)

            if i+1 == self.depth:
                output['reconstruction'] = out

        return output

    def _update_mu_log_var(self, mu_p, log_var_p, mu_q, log_var_q):

        sigma_q = log_var_q.exp()
        sigma_p = log_var_p.exp()
        
        mu_new = ( mu_p / (sigma_p - 1e-6) + mu_q / (sigma_q + 1e-6) ) / ((1 / sigma_p + 1e-6) + (1 / sigma_q + 1e-6))

        log_var_new = (1 / (1 / (sigma_p + 1e-6) + 1 / (sigma_q  + 1e-6))).log()

        return (mu_new, log_var_new)

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps 


class Discriminator_MNIST(BaseDiscriminator):
    """
    A Convolutional encoder Neural net suited for MNIST and Variational Autoencoder-based 
    models.


    It can be built as follows:

    .. code-block::

            >>> from pythae.models.nn.benchmarks.mnist import Discriminator_MNIST
            >>> from pythae.models import VAEGANConfig
            >>> model_config = VAEGANConfig(input_dim=(1, 28, 28), latent_dim=16)
            >>> discriminator = Discriminator_MNIST(model_config)
            >>> discriminator
            ... Discriminator_MNIST(
            ...   (layers): ModuleList(
            ...     (0): Sequential(
            ...       (0): Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): ReLU()
            ...     )
            ...     (1): Sequential(
            ...       (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): Tanh()
            ...     )
            ...     (2): Sequential(
            ...       (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): ReLU()
            ...     )
            ...     (3): Sequential(
            ...       (0): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...       (1): ReLU()
            ...     )
            ...     (4): Sequential(
            ...       (0): Linear(in_features=1024, out_features=1, bias=True)
            ...       (1): Sigmoid()
            ...     )
            ...   )
            ... )

    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAEGAN
        >>> model = VAEGAN(model_config=model_config, discriminator=discriminator)
        >>> model.discriminator == discriminator
        ... True
    """

    def __init__(self, args: dict):
        BaseDiscriminator.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 1
        
        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                #nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1),
                #nn.BatchNorm2d(256),
                nn.Tanh(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1),
                #nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1),
                #nn.BatchNorm2d(1024),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Linear(1024, 1),
                nn.Sigmoid()
            )
        )

        self.layers = layers
        self.depth = len(layers)


    def forward(self, x:torch.Tensor, output_layer_levels:List[int]=None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.
        
        Returns:
            ModelOutput: An instance of ModelOutput containing the adversarial score of the input  
            under the key `embedding`. Optional: The outputs of the layers specified in 
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level.
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

        out = x

        for i in range(max_depth):

            if i == 4:
                out = out.reshape(x.shape[0], -1)
    
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'embedding_layer_{i+1}'] = out
            if i+1 == self.depth:
                output['embedding'] = out

        return output