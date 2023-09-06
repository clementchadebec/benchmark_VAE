import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..normalizing_flows import (
    PlanarFlow,
    PlanarFlowConfig,
    RadialFlow,
    RadialFlowConfig,
)
from ..vae import VAE
from .vae_lin_nf_config import VAE_LinNF_Config


class VAE_LinNF(VAE):
    """Variational Auto Encoder with linear Normalizing Flows model.

    Args:
        model_config(VAE_LinNF_Config): The Variational Autoencoder configuration seting the main
        parameters of the model

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        flows (List[str]): A list of strings corresponding to the class of each flow to be applied.
            Default: Empty list (no flow is applied).

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: VAE_LinNF_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "VAE_LinNF"

        self.net = []
        for flow in model_config.flows:
            if flow == "Planar":
                flow_config = PlanarFlowConfig(
                    input_dim=(model_config.latent_dim,), activation="tanh"
                )
                self.net.append(PlanarFlow(flow_config))

            elif flow == "Radial":
                flow_config = RadialFlowConfig(input_dim=(model_config.latent_dim,))
                self.net.append(RadialFlow(flow_config))

        self.net = nn.ModuleList(self.net)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE with linear normalizing flows model

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

        z0 = z

        log_abs_det_jac = torch.zeros((z0.shape[0],)).to(z.device)

        for layer in self.net:
            layer_output = layer(z)
            z = layer_output.out
            log_abs_det_jac += layer_output.log_abs_det_jac

        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(
            recon_x, x, mu, log_var, z0, z, log_abs_det_jac
        )

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z0, zk, log_abs_det_jac):

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

        # starting gaussian log-density
        log_prob_z0 = (
            -0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))
        ).sum(dim=1)

        # prior log-density
        log_prob_zk = (-0.5 * torch.pow(zk, 2)).sum(dim=1)

        KLD = log_prob_z0 - log_prob_zk - log_abs_det_jac

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

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
        normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.model_config.latent_dim).to(data.device),
            covariance_matrix=torch.eye(self.model_config.latent_dim).to(data.device),
        )

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

                std = torch.exp(0.5 * log_var)
                z, eps = self._sample_gauss(mu, std)

                z0 = z

                log_abs_det_jac = torch.zeros((z0.shape[0],)).to(z.device)

                for layer in self.net:
                    layer_output = layer(z)
                    z = layer_output.out
                    log_abs_det_jac += layer_output.log_abs_det_jac

                log_q_z_given_x = (
                    -0.5 * (log_var + torch.pow(z0 - mu, 2) / torch.exp(log_var))
                ).sum(dim=1) - log_abs_det_jac
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())

        return np.mean(log_p)
