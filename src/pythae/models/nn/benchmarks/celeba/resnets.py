"""Proposed residual neural nets architectures suited for CELEBA"""

from typing import List

import torch
import torch.nn as nn

from ....base import BaseAEConfig
from ....base.base_utils import ModelOutput
from ...base_architectures import BaseDecoder, BaseEncoder
from ..utils import ResBlock


class Encoder_ResNet_AE_CELEBA(BaseEncoder):
    """
    A ResNet encoder suited for CELEBA and Autoencoder-based models.

    It can be built as follows:

    .. code-block::

        >>> from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_AE_CELEBA
        >>> from pythae.models import AEConfig
        >>> model_config = AEConfig(input_dim=(3, 64, 64), latent_dim=16)
        >>> encoder = Encoder_ResNet_AE_CELEBA(model_config)
        >>> encoder
        ... Encoder_ResNet_AE_CELEBA(
        ...   (layers): ModuleList(
        ...     (0): Sequential(
        ...       (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (1): Sequential(
        ...       (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (2): Sequential(
        ...       (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (3): Sequential(
        ...       (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (4): Sequential(
        ...       (0): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (1): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...     )
        ...   )
        ...   (embedding): Linear(in_features=2048, out_features=16, bias=True)
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
            >>> input = torch.rand(2, 3, 64, 64)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 16])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128 * 4 * 4, args.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
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
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))

        return output


class Encoder_ResNet_VAE_CELEBA(BaseEncoder):
    """
    A ResNet encoder suited for CELEBA and Variational Autoencoder-based models.

    It can be built as follows:

    .. code-block::

        >>> from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_VAE_CELEBA
        >>> from pythae.models import VAEConfig
        >>> model_config = VAEConfig(input_dim=(3, 64, 64), latent_dim=16)
        >>> encoder = Encoder_ResNet_VAE_CELEBA(model_config)
        >>> encoder
        ... Encoder_ResNet_VAE_CELEBA(
        ...   (layers): ModuleList(
        ...     (0): Sequential(
        ...       (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (1): Sequential(
        ...       (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (2): Sequential(
        ...       (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (3): Sequential(
        ...       (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (4): Sequential(
        ...       (0): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (1): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...     )
        ...   )
        ...   (embedding): Linear(in_features=2048, out_features=16, bias=True)
        ...   (log_var): Linear(in_features=2048, out_features=16, bias=True)
        ... )

    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VAE
        >>> model = VAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True

    .. note::

        Please note that this encoder is only suitable for Autoencoder based models since it only
        outputs the embeddings of the input data under the key `embedding`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 3, 64, 64)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 16])
    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128 * 4 * 4, args.latent_dim)
        self.log_var = nn.Linear(128 * 4 * 4, args.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
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
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

        return output


class Encoder_ResNet_SVAE_CELEBA(BaseEncoder):
    """
    A ResNet encoder suited for CELEBA and Hyperspherical VAE models.

    It can be built as follows:

    .. code-block::

        >>> from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_SVAE_CELEBA
        >>> from pythae.models import SVAEConfig
        >>> model_config = SVAEConfig(input_dim=(3, 64, 64), latent_dim=16)
        >>> encoder = Encoder_ResNet_SVAE_CELEBA(model_config)
        >>> encoder
        ... Encoder_ResNet_SVAE_CELEBA(
        ...   (layers): ModuleList(
        ...     (0): Sequential(
        ...       (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (1): Sequential(
        ...       (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (2): Sequential(
        ...       (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (3): Sequential(
        ...       (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (4): Sequential(
        ...       (0): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (1): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...     )
        ...   )
        ...   (embedding): Linear(in_features=2048, out_features=16, bias=True)
        ...   (log_concentration): Linear(in_features=2048, out_features=1, bias=True)
        ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import SVAE
        >>> model = SVAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True

    .. note::

        Please note that this encoder is only suitable for Autoencoder based models since it only
        outputs the embeddings of the input data under the key `embedding`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 3, 64, 64)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 16])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128 * 4 * 4, args.latent_dim)
        self.log_concentration = nn.Linear(128 * 4 * 4, 1)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
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
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_concentration"] = self.log_concentration(
                    out.reshape(x.shape[0], -1)
                )

        return output


class Encoder_ResNet_VQVAE_CELEBA(BaseEncoder):
    """
    A ResNet encoder suited for CELEBA and Vector Quantized VAE models.

    It can be built as follows:

    .. code-block::

        >>> from pythae.models.nn.benchmarks.celeba import Encoder_ResNet_VQVAE_CELEBA
        >>> from pythae.models import VQVAEConfig
        >>> model_config = VQVAEConfig(input_dim=(3, 64, 64), latent_dim=16)
        >>> encoder = Encoder_ResNet_VQVAE_CELEBA(model_config)
        >>> encoder
        ... Encoder_ResNet_VQVAE_CELEBA(
        ...   (layers): ModuleList(
        ...     (0): Sequential(
        ...       (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (1): Sequential(
        ...       (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (2): Sequential(
        ...       (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (3): Sequential(
        ...       (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (4): Sequential(
        ...       (0): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (1): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (2): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (3): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...     )
        ...   )
        ...   (pre_qantized): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
        ... )

    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VQVAE
        >>> model = VQVAE(model_config=model_config, encoder=encoder)
        >>> model.encoder == encoder
        ... True

    .. note::

        Please note that this encoder is only suitable for Autoencoder based models since it only
        outputs the embeddings of the input data under the key `embedding`.

        .. code-block::

            >>> import torch
            >>> input = torch.rand(2, 3, 64, 64)
            >>> out = encoder(input)
            >>> out.embedding.shape
            ... torch.Size([2, 16, 4,  4])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.pre_qantized = nn.Conv2d(128, self.latent_dim, 1, 1)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
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
                output["embedding"] = self.pre_qantized(out)

        return output


class Decoder_ResNet_AE_CELEBA(BaseDecoder):
    """
    A ResNet decoder suited for CELEBA and Autoencoder-based
    models.

    .. code-block::

        >>> from pythae.models.nn.benchmarks.celeba import Decoder_ResNet_AE_CELEBA
        >>> from pythae.models import VAEConfig
        >>> model_config = VAEConfig(input_dim=(3, 64, 64), latent_dim=16)
        >>> decoder = Decoder_ResNet_AE_CELEBA(model_config)
        >>> decoder
        ... Decoder_ResNet_AE_CELEBA(
        ...   (layers): ModuleList(
        ...     (0): Linear(in_features=16, out_features=2048, bias=True)
        ...     (1): ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     (2): Sequential(
        ...       (0): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (1): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...     )
        ...     (3): Sequential(
        ...       (0): ConvTranspose2d(128, 128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
        ...       (1): Sigmoid()
        ...     )
        ...     (4): Sequential(
        ...       (0): ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        ...     )
        ...     (5): Sequential(
        ...       (0): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
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
            ... torch.Size([2, 3, 64, 64])
    """

    def __init__(self, args: BaseAEConfig):
        BaseDecoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Linear(args.latent_dim, 128 * 4 * 4))

        layers.append(nn.ConvTranspose2d(128, 128, 3, 2, padding=1))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )

        layers.append(
            nn.Sequential(nn.ConvTranspose2d(128, 128, 5, 2, padding=1), nn.Sigmoid())
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 5, 2, padding=1, output_padding=1)
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, self.n_channels, 4, 2, padding=1), nn.Sigmoid()
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
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

            if i == 0:
                out = out.reshape(z.shape[0], 128, 4, 4)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output


class Decoder_ResNet_VQVAE_CELEBA(BaseDecoder):
    """
    A ResNet decoder suited for CELEBA and Vector Quantized VAE models.

    .. code-block::

        >>> from pythae.models.nn.benchmarks.celeba import Decoder_ResNet_VQVAE_CELEBA
        >>> from pythae.models import VQVAEConfig
        >>> model_config = VQVAEConfig(input_dim=(3, 64, 64), latent_dim=16)
        >>> decoder = Decoder_ResNet_VQVAE_CELEBA(model_config)
        >>> decoder
        ... Decoder_ResNet_VQVAE_CELEBA(
        ...   (layers): ModuleList(
        ...     (0): ConvTranspose2d(16, 128, kernel_size=(1, 1), stride=(1, 1))
        ...     (1): ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ...     (2): Sequential(
        ...       (0): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (1): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (2): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...       (3): ResBlock(
        ...         (conv_block): Sequential(
        ...           (0): ReLU()
        ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ...           (2): ReLU()
        ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
        ...         )
        ...       )
        ...     )
        ...     (3): Sequential(
        ...       (0): ConvTranspose2d(128, 128, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
        ...     )
        ...     (4): Sequential(
        ...       (0): ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        ...     )
        ...     (5): Sequential(
        ...       (0): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ...       (1): Sigmoid()
        ...     )
        ...   )
        ... )


    and then passed to a :class:`pythae.models` instance

        >>> from pythae.models import VQVAE
        >>> model = VQVAE(model_config=model_config, decoder=decoder)
        >>> model.decoder == decoder
        ... True

    .. note::

        Please note that this decoder is suitable for **all** models.

        .. code-block::

            >>> import torch
            >>> input = torch.randn(2, 16, 4, 4)
            >>> out = decoder(input)
            >>> out.reconstruction.shape
            ... torch.Size([2, 3, 64, 64])
    """

    def __init__(self, args: BaseAEConfig):
        BaseDecoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.ConvTranspose2d(self.latent_dim, 128, 1, 1))

        layers.append(nn.ConvTranspose2d(128, 128, 3, 2, padding=1))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )

        layers.append(nn.Sequential(nn.ConvTranspose2d(128, 128, 5, 2, padding=1)))

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 5, 2, padding=1, output_padding=1)
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, self.n_channels, 4, 2, padding=1), nn.Sigmoid()
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
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
