from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..vae import VAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from .sq_vae_config import SQVAEConfig
from .sq_vae_utils import Quantizer


class SQVAE(VAE):
    r"""
    Stochastically Quantized-VAE model.

    Args:
        model_config (SQVAEConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: SQVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self._set_quantizer(model_config)

        self.model_name = "SQVAE"

    def _set_quantizer(self, model_config):

        if model_config.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of VQVAEConfig instance must be set to 'data_shape' where "
                "the shape of the data is (C, H, W ..). Unable to set quantizer."
            )

        x = torch.randn((2,) + self.model_config.input_dim)
        z = self.encoder(x).embedding
        if len(z.shape) == 2:
            z = z.reshape(z.shape[0], 1, 1, -1)

        z = z.permute(0, 2, 3, 1)

        self.model_config.embedding_dim = z.shape[-1]
        self.quantizer = Quantizer(model_config=model_config)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        epoch = kwargs.pop("epoch", 0)

        temperarure = np.max([self.model_config.temperature_init * np.exp(-self.model_config.temperature_decay*epoch), 0])

        encoder_output = self.encoder(x)

        embeddings, log_var = encoder_output.embedding, encoder_output.log_covariance
        var = log_var.exp()


        reshape_for_decoding = False

        if len(embeddings.shape) == 2:
            embeddings = embeddings.reshape(embeddings.shape[0], 1, 1, -1)
            var = var.reshape(embeddings.shape[0], 1, 1, -1)
            reshape_for_decoding = True

        embeddings = embeddings.permute(0, 2, 3, 1)

        quantizer_output = self.quantizer(embeddings, var, temperarure)

        quantized_embed = quantizer_output.quantized_vector
        quantized_indices = quantizer_output.quantized_indices

        if reshape_for_decoding:
            quantized_embed = quantized_embed.reshape(embeddings.shape[0], -1)

        recon_x = self.decoder(quantized_embed).reconstruction

        loss, recon_loss, vq_loss = self.loss_function(recon_x, x, quantizer_output)

        output = ModelOutput(
            recon_loss=recon_loss,
            vq_loss=vq_loss,
            loss=loss,
            recon_x=recon_x,
            z=quantized_embed,
            quantized_indices=quantized_indices,
        )

        return output

    def loss_function(self, recon_x, x, quantizer_output):

        recon_loss = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)

        vq_loss = quantizer_output.loss

        return (
            (recon_loss + vq_loss).mean(dim=0),
            recon_loss.mean(dim=0),
            vq_loss.mean(dim=0),
        )
