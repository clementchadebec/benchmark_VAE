import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..vae import VAE
from .hvae_config import HVAEConfig


class HVAE(VAE):
    r"""
    Hamiltonian VAE.

    Args:
        model_config (HVAEConfig): A model configuration setting the main parameters of the model.

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
        model_config: HVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "HVAE"

        self.n_lf = model_config.n_lf
        self.eps_lf = nn.Parameter(
            torch.tensor([model_config.eps_lf]),
            requires_grad=True if model_config.learn_eps_lf else False,
        )
        self.beta_zero_sqrt = nn.Parameter(
            torch.tensor([model_config.beta_zero]) ** 0.5,
            requires_grad=True if model_config.learn_beta_zero else False,
        )

        if model_config.reconstruction_loss == "bce":
            warnings.warn(
                "Carefull, this model expects the encoder to give the *logits* of the Bernouilli "
                "distribution. Make sure the encoder actually outputs the logits."
            )

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        r"""
        The input data is first encoded. The reparametrization is used to produce a sample
        :math:`z_0` from the approximate posterior :math:`q_{\phi}(z|x)`. Then
        Hamiltonian equations are solved using the leapfrog integrator.

        Args:
            inputs (BaseDataset): The training data with labels

        Returns:
            output (ModelOutput): An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = F.softplus(log_var)
        z0, eps0 = self._sample_gauss(mu, std)
        gamma = torch.randn_like(z0, device=x.device)
        rho = gamma / self.beta_zero_sqrt

        z = z0
        beta_sqrt_old = self.beta_zero_sqrt

        for k in range(self.n_lf):

            # perform leapfrog steps

            # 1st leapfrog step
            rho_ = rho - (self.eps_lf / 2) * self._dU_dz(z, x)

            # 2nd leapfrog step
            z_ = z + self.eps_lf * rho

            # 3rd leapfrog step
            rho__ = rho_ - (self.eps_lf / 2) * self._dU_dz(z_, x)

            # tempering steps
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

            z = z_
            rho = rho__

        recon_x = self.decoder(z)["reconstruction"].reshape_as(x)

        loss = self.loss_function(x, z, rho, z0, mu, log_var)

        output = ModelOutput(
            loss=loss,
            recon_x=recon_x,
            z=z,
            z0=z0,
            rho=rho,
            eps0=eps0,
            gamma=gamma,
            mu=mu,
            log_var=log_var,
        )

        return output

    def _dU_dz(self, z, x):
        net_out = self.decoder(z)["reconstruction"].reshape(x.shape[0], -1)
        U = -self._log_p_x_given_z(net_out, x).sum()

        g = grad(U, z)[0]

        return g + z

    def loss_function(self, x, zK, rhoK, z0, mu, log_var):

        recon_x = self.decoder(zK)["reconstruction"]

        logpx_given_z = self._log_p_x_given_z(recon_x, x)  # log p(x|z_K)

        log_zk = -0.5 * torch.pow(zK, 2).sum(dim=-1)  # log p(\z_K)
        logrhoK = -0.5 * torch.pow(rhoK, 2).sum(dim=-1)  # log p(\rho_K)
        logp = logpx_given_z + logrhoK + log_zk

        logq = -0.5 * log_var.sum(
            dim=-1
        )  # (-0.5 * (log_var + torch.pow(z0 - mu, 2) / log_var.exp())).sum(dim=1)  # q(z_0|x)

        return -(logp - logq).mean(dim=0)

    def _tempering(self, k, K):
        """Perform tempering step"""

        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k

    def _log_p_x_given_z(self, recon_x, x):

        if self.model_config.reconstruction_loss == "mse":
            # sigma is taken as I_D
            logp_x_given_z = (
                -0.5
                * F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1)
            )
            # - torch.log(torch.tensor([2 * np.pi]).to(x.device)) \
            #    * np.prod(self.input_dim) / 2

        elif self.model_config.reconstruction_loss == "bce":

            logp_x_given_z = (
                torch.distributions.Bernoulli(logits=recon_x.reshape(x.shape[0], -1))
                .log_prob(x.reshape(x.shape[0], -1))
                .sum(dim=-1)
            )

        return logp_x_given_z

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

                x_rep = torch.cat(batch_size * [x]).reshape(-1, 1, 28, 28)

                encoder_output = self.encoder(x_rep)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance

                std = torch.exp(0.5 * log_var)
                z0, _ = self._sample_gauss(mu, std)
                gamma = torch.randn_like(z0, device=x.device)
                rho = gamma / self.beta_zero_sqrt

                z = z0
                beta_sqrt_old = self.beta_zero_sqrt

                for k in range(self.n_lf):

                    # 1st leapfrog step
                    rho_ = rho - (self.eps_lf / 2) * self._dU_dz(z, x_rep)

                    # 2nd leapfrog step
                    z = z + self.eps_lf * rho_

                    # 3rd leapfrog step
                    rho__ = rho_ - (self.eps_lf / 2) * self._dU_dz(z, x_rep)

                    # tempering steps
                    beta_sqrt = self._tempering(k + 1, self.n_lf)
                    rho = (beta_sqrt_old / beta_sqrt) * rho__
                    beta_sqrt_old = beta_sqrt

                log_q_z0_given_x = -0.5 * (
                    log_var + (z0 - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)
                log_p_rho = -0.5 * (rho ** 2).sum(dim=-1)

                log_p_rho0 = -0.5 * (rho ** 2).sum(dim=-1) * self.beta_zero_sqrt

                recon_x = self.decoder(z)["reconstruction"]

                log_p_x_given_z = self._log_p_x_given_z(recon_x, x_rep)

                log_p_x.append(
                    log_p_x_given_z
                    + log_p_z
                    + log_p_rho
                    - log_p_rho0
                    - log_q_z0_given_x
                )  # N*log(2*pi) simplifies in prior and posterior

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())

            if i % 50 == 0:
                print(f"Current nll at {i}: {np.mean(log_p)}")

        return np.mean(log_p)
