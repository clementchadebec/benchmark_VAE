"""Proposed neural nets architectures suited for MNIST"""

import torch
import torch.nn as nn
import numpy as np

from typing import List
from ..base_architectures import BaseEncoder, BaseDecoder
from ....models.base.base_utils import ModelOuput
from ....models import BaseAEConfig

from pythae.models.nn import (
    BaseEncoder,
    BaseDecoder,
    BaseLayeredDiscriminator
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
            ...     (conv_layers): Sequential(
            ...         (0): Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...         (2): ReLU()
            ...         (3): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...         (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...         (5): ReLU()
            ...         (6): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...         (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...         (8): ReLU()
            ...         (9): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...         (10): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...         (11): ReLU()
            ...     )
            ...         (embedding): Linear(in_features=1024, out_features=10, bias=True)
            ...     )

    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import AE
        >>> model = AE(model_config=model_config, encoder=encoder)
        >>> model.encoder
        ... Encoder_AE_MNIST(
        ...   (conv_layers): Sequential(
        ...     (0): Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (2): ReLU()
        ...     (3): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (5): ReLU()
        ...     (6): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (8): ReLU()
        ...     (9): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (10): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (11): ReLU()
        ...   )
        ...   (embedding): Linear(in_features=1024, out_features=16, bias=True)
        ... )

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
        
        Returns:
            ModelOuput: An instance of ModelOutput containing the embeddings of the input data under
            the key `embedding`"""
        output = ModelOuput()

        if output_layer_levels is not None:

            assert all(self.depth >= levels > 0), (
                f'Cannot output layer deeper than depth ({self.depth}) or with non-positive indice. '\
                f'Got ({output_layer_levels})'
                )

        out = x

        for i in range(self.depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'embedding_layer_{i+1}'] = out
        
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
            ...   (conv_layers): Sequential(
            ...     (0): Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...     (2): ReLU()
            ...     (3): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...     (5): ReLU()
            ...     (6): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...     (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...     (8): ReLU()
            ...     (9): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...     (10): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...     (11): ReLU()
            ...   )
            ...   (embedding): Linear(in_features=1024, out_features=16, bias=True)
            ...   (log_var): Linear(in_features=1024, out_features=16, bias=True)
            ... )

    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, encoder=encoder)
        >>> model.encoder
        ... Encoder_VAE_MNIST(
        ...   (conv_layers): Sequential(
        ...     (0): Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (2): ReLU()
        ...     (3): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (5): ReLU()
        ...     (6): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (8): ReLU()
        ...     (9): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (10): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (11): ReLU()
        ...   )
        ...   (embedding): Linear(in_features=1024, out_features=16, bias=True)
        ...   (log_var): Linear(in_features=1024, out_features=16, bias=True)
        ... )

    .. note::

        Please note that this encoder is only suitable for Variational Autoencoder based models 
        since it outputs the embeddings and the **log** of the covariance diagonal coffecients 
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
        
        Returns:
            ModelOuput: An instance of ModelOutput containing the embeddings of the input data under
            the key `embedding` and the **log** of the diagonal coefficient of the covariance 
            matrices under the key `log_covariance`"""
        output = ModelOuput()

        if output_layer_levels is not None:

            assert all(self.depth >= levels > 0), (
                f'Cannot output layer deeper than depth ({self.depth}) or with non-positive indice. '\
                f'Got ({output_layer_levels})'
                )

        out = x

        for i in range(self.depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'embedding_layer_{i+1}'] = out
        
        output['embedding'] = self.embedding(out.reshape(x.shape[0], -1))
        output['log_covariance'] = self.log_var(out.reshape(x.shape[0], -1))

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
            ...   (fc): Linear(in_features=16, out_features=16384, bias=True)
            ...   (deconv_layers): Sequential(
            ...     (0): ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            ...     (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...     (2): ReLU()
            ...     (3): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
            ...     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...     (5): ReLU()
            ...     (6): ConvTranspose2d(256, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
            ...     (7): Sigmoid()
            ...   )
            ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, decoder=decoder)
        >>> model.decoder
        ... Decoder_AE_MNIST(
        ...   (fc): Linear(in_features=16, out_features=16384, bias=True)
        ...   (deconv_layers): Sequential(
        ...     (0): ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (2): ReLU()
        ...     (3): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        ...     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (5): ReLU()
        ...     (6): ConvTranspose2d(256, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        ...     (7): Sigmoid()
        ...   )
        ... )

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
        
        Returns:
            ModelOuput: An instance of ModelOutput containing the reconstruction of the latent code 
            under the key `reconstruction`
        """
        output = ModelOuput()

        if output_layer_levels is not None:

            assert all(self.depth >= levels > 0), (
                f'Cannot output layer deeper than depth ({self.depth}) or with non-positive indice. '\
                f'Got ({output_layer_levels})'
                )

        out = z

        for i in range(self.depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'reconstruction_layer_{i+1}'] = out

            if i == 0:
                out = out.reshape(z.shape[0], 1024, 4, 4)

        output['reconstruction'] = out

        return output


class LayeredDiscriminator_MNIST(BaseLayeredDiscriminator):
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
            ...   (conv_layers): Sequential(
            ...     (0): Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...     (2): ReLU()
            ...     (3): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...     (5): ReLU()
            ...     (6): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...     (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...     (8): ReLU()
            ...     (9): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            ...     (10): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ...     (11): ReLU()
            ...   )
            ...   (embedding): Linear(in_features=1024, out_features=16, bias=True)
            ...   (log_var): Linear(in_features=1024, out_features=16, bias=True)
            ... )

    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, encoder=encoder)
        >>> model.encoder
        ... Encoder_VAE_MNIST(
        ...   (conv_layers): Sequential(
        ...     (0): Conv2d(1, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (2): ReLU()
        ...     (3): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (5): ReLU()
        ...     (6): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (7): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (8): ReLU()
        ...     (9): Conv2d(512, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     (10): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ...     (11): ReLU()
        ...   )
        ...   (embedding): Linear(in_features=1024, out_features=16, bias=True)
        ...   (log_var): Linear(in_features=1024, out_features=16, bias=True)
        ... )
    """

    def __init__(self, args: dict):

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

        BaseLayeredDiscriminator.__init__(self, layers=layers)

    def forward(self, x:torch.Tensor, output_layer_level:int=None):

        if output_layer_level is not None:

            assert output_layer_level <= self.depth, (
                f'Cannot output layer deeper ({output_layer_level}) than depth ({self.depth})'
            )

        for i in range(self.depth):

            if i == 4:
                x = x.reshape(x.shape[0], -1)

            x = self.layers[i](x)

            if i + 1 == output_layer_level:
                break
        
        output = ModelOuput(
            adversarial_cost=x
        )
    
        return output