import os
from typing import Optional

import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..vae import VAE
from .info_vae_config import INFOVAE_MMD_Config


class INFOVAE_MMD(VAE):
    """Info Variational Autoencoder model.

    Args:
        model_config (INFOVAE_MMD_Config): The Autoencoder configuration setting the main
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
        model_config: INFOVAE_MMD_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "INFOVAE_MMD"

        self.alpha = self.model_config.alpha
        self.lbd = self.model_config.lbd
        self.kernel_choice = model_config.kernel_choice
        self.scales = model_config.scales if model_config.scales is not None else [1.0]

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        z_prior = torch.randn_like(z, device=x.device)

        loss, recon_loss, kld_loss, mmd_loss = self.loss_function(
            recon_x, x, z, z_prior, mu, log_var
        )

        output = ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            reg_loss=kld_loss,
            mmd_loss=mmd_loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, z, z_prior, mu, log_var):

        N = z.shape[0]  # batch size

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = (
                0.5
                * F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1)
            )

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        if self.kernel_choice == "rbf":
            k_z = self.rbf_kernel(z, z)
            k_z_prior = self.rbf_kernel(z_prior, z_prior)
            k_cross = self.rbf_kernel(z, z_prior)

        else:
            k_z = self.imq_kernel(z, z)
            k_z_prior = self.imq_kernel(z_prior, z_prior)
            k_cross = self.imq_kernel(z, z_prior)

        mmd_z = (k_z - k_z.diag().diag()).sum() / ((N - 1) * N)
        mmd_z_prior = (k_z_prior - k_z_prior.diag().diag()).sum() / ((N - 1) * N)
        mmd_cross = k_cross.sum() / (N ** 2)

        mmd_loss = mmd_z + mmd_z_prior - 2 * mmd_cross

        loss = (
            recon_loss + (1 - self.alpha) * KLD + (self.alpha + self.lbd - 1) * mmd_loss
        )

        return (
            (loss).mean(dim=0),
            (recon_loss).mean(dim=0),
            (KLD).mean(dim=0),
            mmd_loss,
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def imq_kernel(self, z1, z2):
        """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""

        Cbase = (
            2.0 * self.model_config.latent_dim * self.model_config.kernel_bandwidth ** 2
        )

        k = 0

        for scale in self.scales:
            C = scale * Cbase
            k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)

        return k

    def rbf_kernel(self, z1, z2):
        """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""

        C = 2.0 * self.model_config.latent_dim * self.model_config.kernel_bandwidth ** 2

        k = torch.exp(-torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C)

        return k
