import os
from typing import Optional

import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..ae import AE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from .rae_l2_config import RAE_L2_Config


class RAE_L2(AE):
    """Regularized Autoencoder with L2 decoder params regularization model.

    Args:
        model_config (RAE_L2_Config): The Autoencoder configuration setting the main parameters of the
            model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: RAE_L2_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        AE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "RAE_L2"

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]

        z = self.encoder(x).embedding
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, embedding_loss = self.loss_function(recon_x, x, z)

        output = ModelOutput(
            loss=loss,
            encoder_loss=loss,
            decoder_loss=loss,
            update_encoder=True,
            update_decoder=True,
            recon_loss=recon_loss,
            embedding_loss=embedding_loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, z):

        recon_loss = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)

        embedding_loss = 0.5 * torch.linalg.norm(z, dim=-1) ** 2

        return (
            (recon_loss + self.model_config.embedding_weight * embedding_loss).mean(
                dim=0
            ),
            (recon_loss).mean(dim=0),
            (embedding_loss).mean(dim=0),
        )
