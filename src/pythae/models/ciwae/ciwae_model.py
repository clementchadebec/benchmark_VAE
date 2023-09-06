from typing import Optional

import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..vae import VAE
from .ciwae_config import CIWAEConfig


class CIWAE(VAE):
    """
    Combination Importance Weighted Autoencoder model.

    Args:
        model_config (CIWAEConfig): The CIWAE configuration setting the main
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
        model_config: CIWAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "CIWAE"
        self.n_samples = model_config.number_samples
        self.beta = model_config.beta

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

        mu = mu.unsqueeze(1).repeat(1, self.n_samples, 1)
        log_var = log_var.unsqueeze(1).repeat(1, self.n_samples, 1)

        std = torch.exp(0.5 * log_var)

        z, _ = self._sample_gauss(mu, std)

        recon_x = self.decoder(z.reshape(-1, self.latent_dim))[
            "reconstruction"
        ].reshape(x.shape[0], self.n_samples, -1)

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x.reshape(x.shape[0], self.n_samples, -1)[:, 0, :].reshape_as(
                x
            ),
            z=z.reshape(x.shape[0], self.n_samples, -1)[:, 0, :].reshape(
                -1, self.latent_dim
            ),
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = (
                0.5
                * F.mse_loss(
                    recon_x,
                    x.reshape(recon_x.shape[0], -1)
                    .unsqueeze(1)
                    .repeat(1, self.n_samples, 1),
                    reduction="none",
                ).sum(dim=-1)
            )

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x,
                x.reshape(recon_x.shape[0], -1)
                .unsqueeze(1)
                .repeat(1, self.n_samples, 1),
                reduction="none",
            ).sum(dim=-1)

        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=-1)
        log_p_z = -0.5 * (z ** 2).sum(dim=-1)

        KLD = -(log_p_z - log_q_z)

        log_w = -(recon_loss + KLD)

        log_w_minus_max = log_w - log_w.max(1, keepdim=True)[0]
        w = log_w_minus_max.exp()
        w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()

        iwae_elbo = -(w_tilde * log_w).sum(1)
        vae_elbo = -log_w.mean(dim=-1)  # avg on importance samples

        return (
            (self.beta * vae_elbo + (1 - self.beta) * iwae_elbo).mean(dim=0),
            recon_loss.mean(dim=0),
            KLD.mean(dim=0),
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps
