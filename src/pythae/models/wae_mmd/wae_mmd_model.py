import os
from typing import Optional

import torch
import torch.nn.functional as F
from pyexpat import model

from ...data.datasets import BaseDataset
from ..ae import AE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from .wae_mmd_config import WAE_MMD_Config


class WAE_MMD(AE):
    """Wasserstein Autoencoder model.

    Args:
        model_config (WAE_MMD_Config): The Autoencoder configuration setting the main parameters
            of the model.

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
        model_config: WAE_MMD_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        AE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "WAE_MMD"

        self.kernel_choice = model_config.kernel_choice
        self.scales = model_config.scales if model_config.scales is not None else [1.0]
        self.reconstruction_loss_scale = self.model_config.reconstruction_loss_scale

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

        z_prior = torch.randn_like(z, device=x.device)

        loss, recon_loss, mmd_loss = self.loss_function(recon_x, x, z, z_prior)

        output = ModelOutput(
            loss=loss, recon_loss=recon_loss, mmd_loss=mmd_loss, recon_x=recon_x, z=z
        )

        return output

    def loss_function(self, recon_x, x, z, z_prior):

        N = z.shape[0]  # batch size

        recon_loss = self.reconstruction_loss_scale * F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)

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

        return (
            recon_loss.mean(dim=0) + self.model_config.reg_weight * mmd_loss,
            (recon_loss).mean(dim=0),
            mmd_loss,
        )

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
