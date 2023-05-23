import os
from typing import Optional

import torch

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..vae import VAE
from .msssim_vae_config import MSSSIM_VAEConfig
from .msssim_vae_utils import MSSSIM


class MSSSIM_VAE(VAE):
    r"""
    VAE using perseptual similarity metrics model.

    Args:
        model_config (MSSSIM_VAEConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

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
        model_config: MSSSIM_VAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "MSSSIM_VAE"
        self.beta = model_config.beta
        self.msssim = MSSSIM(window_size=model_config.window_size)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z):

        recon_loss = self.msssim(recon_x, x)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return (
            (recon_loss + self.beta * KLD).mean(dim=0),
            recon_loss.mean(dim=0),
            KLD.mean(dim=0),
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps
