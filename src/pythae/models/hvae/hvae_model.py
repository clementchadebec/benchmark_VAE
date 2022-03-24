import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from ...data.datasets import BaseDataset
from ...models import VAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from .hvae_config import HVAEConfig


class HVAE(VAE):
    r"""
    Hamiltonian VAE.

    Args:
        model_config (HVAEConfig): A model configuration setting the main parameters of the model

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

        std = torch.exp(0.5 * log_var)
        z0, eps0 = self._sample_gauss(mu, std)
        gamma = torch.randn_like(z0, device=x.device)
        rho = gamma / self.beta_zero_sqrt

        z = z0
        beta_sqrt_old = self.beta_zero_sqrt

        recon_x = self.decoder(z)["reconstruction"]

        for k in range(self.n_lf):

            # perform leapfrog steps

            # computes potential energy
            U = -self._log_p_xz(recon_x, x, z).sum()

            # Compute its gradient
            g = grad(U, z, create_graph=True)[0]

            # 1st leapfrog step
            rho_ = rho - (self.eps_lf / 2) * g

            # 2nd leapfrog step
            z = z + self.eps_lf * rho_

            recon_x = self.decoder(z)["reconstruction"]

            U = -self._log_p_xz(recon_x, x, z).sum()
            g = grad(U, z, create_graph=True)[0]

            # 3rd leapfrog step
            rho__ = rho_ - (self.eps_lf / 2) * g

            # tempering steps
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        loss = self.loss_function(recon_x, x, z0, z, rho, eps0, gamma, mu, log_var)

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

    def loss_function(self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var):

        normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.model_config.latent_dim).to(x.device),
            covariance_matrix=torch.eye(self.model_config.latent_dim).to(x.device),
        )

        logpxz = self._log_p_xz(recon_x.reshape(x.shape[0], -1), x, zK)  # log p(x, z_K)
        logrhoK = -0.5 * torch.pow(rhoK, 2).sum(dim=-1)  # log p(\rho_K)
        logp = logpxz + logrhoK

        logq = normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # q(z_0|x)

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
            recon_loss = -0.5 * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
            # - torch.log(torch.tensor([2 * np.pi]).to(x.device)) \
            #    * np.prod(self.input_dim) / 2

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = -F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        return recon_loss

    def _log_z(self, z):
        """
        Return Normal density function as prior on z
        """

        # define a N(0, I) distribution
        normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim).to(z.device),
            covariance_matrix=torch.eye(self.latent_dim).to(z.device),
        )
        return -0.5 * torch.pow(z, 2).sum(dim=-1)

    def _log_p_xz(self, recon_x, x, z):
        """
        Estimate log(p(x, z)) using Bayes rule
        """
        logpxz = self._log_p_x_given_z(recon_x, x)
        logpz = self._log_z(z)
        return logpxz + logpz

    def _hamiltonian(self, recon_x, x, z, rho, G_inv=None, G_log_det=None):
        """
        Computes the Hamiltonian function.
        used for HVAE
        """
        return -self.log_p_xz(recon_x.reshape(x.shape[0], -1), x, z).sum()

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior disctribution. This may take a while.

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
                z0, eps = self._sample_gauss(mu, std)
                gamma = torch.randn_like(z0, device=x.device)
                rho = gamma / self.beta_zero_sqrt

                z = z0
                beta_sqrt_old = self.beta_zero_sqrt

                recon_x = self.decoder(z)["reconstruction"]

                for k in range(self.n_lf):

                    # perform leapfrog steps

                    # computes potential energy
                    U = -self._log_p_xz(recon_x, x_rep, z).sum()

                    # Compute its gradient
                    g = grad(U, z, create_graph=True)[0]

                    # 1st leapfrog step
                    rho_ = rho - (self.eps_lf / 2) * g

                    # 2nd leapfrog step
                    z = z + self.eps_lf * rho_

                    recon_x = self.decoder(z)["reconstruction"]

                    U = -self._log_p_xz(recon_x, x_rep, z).sum()
                    g = grad(U, z, create_graph=True)[0]

                    # 3rd leapfrog step
                    rho__ = rho_ - (self.eps_lf / 2) * g

                    # tempering steps
                    beta_sqrt = self._tempering(k + 1, self.n_lf)
                    rho = (beta_sqrt_old / beta_sqrt) * rho__
                    beta_sqrt_old = beta_sqrt

                log_q_z0_given_x = -0.5 * (
                    log_var + (z0 - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z**2).sum(dim=-1)

                log_p_rho0 = normal.log_prob(gamma) - 0.5 * self.latent_dim * torch.log(
                    1 / self.beta_zero_sqrt) # rho0 ~ N(0, 1/beta_0*I)
                log_p_rho = normal.log_prob(rho)

                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = -F.mse_loss(
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
                    log_p_x_given_z + log_p_z + log_p_rho - log_p_rho0 - log_q_z0_given_x
                ) # N*log(2*pi) simplifies in prior and posterior

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
            
        return np.mean(log_p)   

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = HVAEConfig.from_json_file(path_to_model_config)

        return model_config

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)

        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)

        else:
            encoder = None

        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)

        else:
            decoder = None

        model = cls(model_config, encoder=encoder, decoder=decoder)
        model.load_state_dict(model_weights)

        return model
