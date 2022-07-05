import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..vae import VAE
from .iwae_config import IWAEConfig


class IWAE(VAE):
    """
    Importance Weighted Autoencoder model.

    Args:
        model_config (IWAEConfig): The IWAE configuration setting the main
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
        model_config: IWAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "IWAE"
        self.n_samples = model_config.number_samples

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
        recon_x = self.decoder(z.reshape(-1, self.latent_dim))["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)

        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x.reshape(x.shape[0], self.n_samples, -1)[:, 0, :].reshape_as(
                x
            ),
            z=z[:, 0, :].reshape(-1, self.latent_dim),
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = (
                F.mse_loss(
                    recon_x.reshape(recon_x.shape[0], -1),
                    x.reshape(x.shape[0], -1)
                    .unsqueeze(1)
                    .repeat(1, self.n_samples, 1)
                    .reshape(recon_x.shape[0], -1),
                    reduction="none",
                )
                .sum(dim=-1)
                .reshape(x.shape[0], -1)
            )

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = (
                F.binary_cross_entropy(
                    recon_x.reshape(recon_x.shape[0], -1),
                    x.reshape(x.shape[0], -1)
                    .unsqueeze(1)
                    .repeat(1, self.n_samples, 1)
                    .reshape(recon_x.shape[0], -1),
                    reduction="none",
                )
                .sum(dim=-1)
                .reshape(x.shape[0], -1)
            )

        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=-1)
        log_p_z = -0.5 * (z ** 2).sum(dim=-1)

        KLD = -(log_p_z - log_q_z)

        log_w = recon_loss + KLD

        w_tilde = F.softmax(log_w.detach(), dim=-1)

        return (
            (w_tilde * log_w).sum(dim=-1).mean(dim=0),
            recon_loss.mean(),
            KLD.mean(),
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):

                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                mu = mu.unsqueeze(1).repeat(1, self.n_samples, 1)
                log_var = log_var.unsqueeze(1).repeat(1, self.n_samples, 1)

                std = torch.exp(0.5 * log_var)
                z, _ = self._sample_gauss(mu, std)

                log_q_z_given_x = -0.5 * (
                    log_var + (z - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)

                recon_x = self.decoder(z.reshape(-1, self.latent_dim))["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(recon_x.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1)
                        .unsqueeze(1)
                        .repeat(1, self.n_samples, 1)
                        .reshape(recon_x.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1).reshape(x_rep.shape[0], -1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = (
                        -F.binary_cross_entropy(
                            recon_x.reshape(recon_x.shape[0], -1),
                            x_rep.reshape(x_rep.shape[0], -1)
                            .unsqueeze(1)
                            .repeat(1, self.n_samples, 1)
                            .reshape(recon_x.shape[0], -1),
                            reduction="none",
                        )
                        .sum(dim=-1)
                        .reshape(x_rep.shape[0], -1)
                    )

                log_w = log_p_x_given_z + log_p_z - log_q_z_given_x

                log_p_x.append((log_w).exp().mean(dim=-1).log())  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)
