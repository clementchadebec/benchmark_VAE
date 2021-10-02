import os
from copy import deepcopy
from typing import Optional

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from pyraug.customexception import BadInheritanceError
from pyraug.models.base.base_utils import ModelOuput
from pyraug.models.base.base_vae import BaseVAE
from pyraug.models.nn import BaseDecoder, BaseEncoder, BaseMetric
from pyraug.models.nn.default_architectures import Metric_MLP
from pyraug.models.rhvae.rhvae_config import RHVAEConfig

from .rhvae_utils import create_inverse_metric, create_metric


class RHVAE(BaseVAE):
    r"""
    This is an implementation of the Riemannian Hamiltonian VAE model proposed in
    (https://arxiv.org/pdf/2010.11518.pdf). This model provides a way to
    learn the Riemannian latent structure of a given set of data set through a parametrized
    Riemannian metric having the following shape:
    :math:`\mathbf{G}^{-1}(z) = \sum \limits _{i=1}^N L_{\psi_i} L_{\psi_i}^{\top} \exp
    \Big(-\frac{\lVert z - c_i \rVert_2^2}{T^2} \Big) + \lambda I_d`

    and to generate new data. It is particularly well suited for High
    Dimensional data combined with low sample number and proved relevant for Data Augmentation as
    proved in (https://arxiv.org/pdf/2105.00026.pdf).


    Args:
        model_config (RHVAEConfig): A model configuration setting the main parameters of the model

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: RHVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        metric: Optional[BaseMetric] = None,
    ):

        BaseVAE.__init__(
            self, model_config=model_config, encoder=encoder, decoder=decoder
        )

        if metric is None:
            metric = Metric_MLP(model_config)
            self.model_config.uses_default_metric = True

        else:
            self.model_config.uses_default_metric = False

        self.set_metric(metric)

        self.temperature = nn.Parameter(
            torch.Tensor([model_config.temperature]), requires_grad=False
        )
        self.lbd = nn.Parameter(
            torch.Tensor([model_config.regularization]), requires_grad=False
        )
        self.beta_zero_sqrt = nn.Parameter(
            torch.Tensor([model_config.beta_zero]), requires_grad=False
        )
        self.n_lf = model_config.n_lf
        self.eps_lf = model_config.eps_lf

        # this is used to store the matrices and centroids throughout training for
        # further use in metric update (L is the cholesky decomposition of M)
        self.M = []
        self.centroids = []

        self.M_tens = torch.randn(
            1, self.model_config.latent_dim, self.model_config.latent_dim
        )
        self.centroids_tens = torch.randn(1, self.model_config.latent_dim)

        # define a starting metric (gamma_i = 0 & L = I_d)
        def G(z):
            return torch.inverse(
                (
                    torch.eye(self.latent_dim, device=z.device).unsqueeze(0)
                    * torch.exp(-torch.norm(z.unsqueeze(1), dim=-1) ** 2)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1)
                + self.lbd * torch.eye(self.latent_dim).to(z.device)
            )

        def G_inv(z):
            return (
                torch.eye(self.latent_dim, device=z.device).unsqueeze(0)
                * torch.exp(-torch.norm(z.unsqueeze(1), dim=-1) ** 2)
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(z.device)

        self.G = G
        self.G_inv = G_inv

    def update(self):
        self.update_metric()

    def set_metric(self, metric: BaseMetric) -> None:
        r"""This method is called to set the metric network outputing the
        :math:`L_{\psi_i}` of the metric matrices

        Args:
            metric (BaseMetric): The metric module that need to be set to the model.

        """
        if not issubclass(type(metric), BaseMetric):
            raise BadInheritanceError(
                (
                    "Metric must inherit from BaseMetric class from "
                    "pyraug.models.base_architectures.BaseMetric. Refer to documentation."
                )
            )

        self.metric = metric

    def forward(self, inputs):
        r"""
        The input data is first encoded. The reparametrization is used to produce a sample
        :math:`z_0` from the approximate posterior :math:`q_{\phi}(z|x)`. Then Riemannian
        Hamiltonian equations are solved using the generalized leapfrog integrator. In the meantime,
        the input data :math:`x` is fed to the metric network outputing the matrices
        :math:`L_{\psi}`. The metric is computed and used with the integrator.

        Args:
            inputs (Dict[str, torch.Tensor]): The training data with labels

        Returns:
            output (ModelOutput): An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]

        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        z0, eps0 = self._sample_gauss(mu, std)

        z = z0

        if self.training:
            # update the metric using batch data points
            L = self.metric(x)

            M = L @ torch.transpose(L, 1, 2)

            # store LL^T and mu(x_i) to update final metric
            self.M.append(M.clone().detach())
            self.centroids.append(mu.clone().detach())

            G_inv = (
                M.unsqueeze(0)
                * torch.exp(
                    -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                    / (self.temperature ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(x.device)

        else:
            G = self.G(z)
            G_inv = self.G_inv(z)
            L = torch.linalg.cholesky(G)

        G_log_det = -torch.logdet(G_inv)

        gamma = torch.randn_like(z0, device=x.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        # sample \rho from N(0, G)
        rho = (L @ rho.unsqueeze(-1)).squeeze(-1)

        recon_x = self.decoder(z)

        for k in range(self.n_lf):

            # perform leapfrog steps

            # step 1
            rho_ = self._leap_step_1(recon_x, x, z, rho, G_inv, G_log_det)

            # step 2
            z = self._leap_step_2(recon_x, x, z, rho_, G_inv, G_log_det)

            recon_x = self.decoder(z)

            if self.training:

                G_inv = (
                    M.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                        / (self.temperature ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(x.device)

            else:
                # compute metric value on new z using final metric
                G = self.G(z)
                G_inv = self.G_inv(z)

            G_log_det = -torch.logdet(G_inv)

            # step 3
            rho__ = self._leap_step_3(recon_x, x, z, rho_, G_inv, G_log_det)

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        loss = self.loss_function(
            recon_x, x, z0, z, rho, eps0, gamma, mu, log_var, G_inv, G_log_det
        )

        output = ModelOuput(
            loss=loss,
            recon_x=recon_x,
            z=z,
            z0=z0,
            rho=rho,
            eps0=eps0,
            gamma=gamma,
            mu=mu,
            log_var=log_var,
            G_inv=G_inv,
            G_log_det=G_log_det,
        )

        return output

    def save(self, dir_path: str):
        """Method to save the model at a specific location

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """

        # This creates the dir if not available
        super().save(dir_path)

        model_path = dir_path

        model_dict = {
            "M": deepcopy(self.M_tens.clone().detach()),
            "centroids": deepcopy(self.centroids_tens.clone().detach()),
            "model_state_dict": deepcopy(self.state_dict()),
        }

        if not self.model_config.uses_default_metric:
            with open(os.path.join(model_path, "metric.pkl"), "wb") as fp:
                dill.dump(self.metric, fp)

        torch.save(model_dict, os.path.join(model_path, "model.pt"))

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = RHVAEConfig.from_json_file(path_to_model_config)

        return model_config

    @classmethod
    def _load_custom_metric_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)

        if "metric.pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing metric pkl file ('metric.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom metrics."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "metric.pkl"), "rb") as fp:
                metric = dill.load(fp)

        return metric

    @classmethod
    def _load_metric_matrices_and_centroids(cls, dir_path):
        """this function can be called safely since it is called after
        _load_model_weights_from_folder which handles FileNotFoundError and
        loading issues"""

        path_to_model_weights = os.path.join(dir_path, "model.pt")

        model_weights = torch.load(path_to_model_weights, map_location="cpu")

        if "M" not in model_weights.keys():
            raise KeyError(
                "Metric M matrices are not available in 'model.pt' file. Got keys:"
                f"{model_weights.keys()}. These are needed to build the metric."
            )

        metric_M = model_weights["M"]

        if "centroids" not in model_weights.keys():
            raise KeyError(
                "Metric centroids are not available in 'model.pt' file. Got keys:"
                f"{model_weights.keys()}. These are needed to build the metric."
            )

        metric_centroids = model_weights["centroids"]

        return metric_M, metric_centroids

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:
                a ``model_config.json`` and a ``model.pt`` if no custom architectures were
                provided
                a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl`` or/and ``metric.pkl``) if a custom encoder (resp. decoder or/and
                metric) was provided
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

        if not model_config.uses_default_metric:
            metric = cls._load_custom_metric_from_folder(dir_path)

        else:
            metric = None

        model = cls(model_config, encoder=encoder, decoder=decoder, metric=metric)

        metric_M, metric_centroids = cls._load_metric_matrices_and_centroids(dir_path)

        model.M_tens = metric_M
        model.centroids_tens = metric_centroids

        model.G = create_metric(model)
        model.G_inv = create_inverse_metric(model)

        model.load_state_dict(model_weights)

        return model

    def _leap_step_1(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves first equation of generalized leapfrog integrator
        using fixed point iterations
        """

        def f_(rho_):
            H = self._hamiltonian(recon_x, x, z, rho_, G_inv, G_log_det)
            gz = grad(H, z, retain_graph=True)[0]
            return rho - 0.5 * self.eps_lf * gz

        rho_ = rho.clone()
        for _ in range(steps):
            rho_ = f_(rho_)
        return rho_

    def _leap_step_2(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves second equation of generalized leapfrog integrator
        using fixed point iterations
        """
        H0 = self._hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        grho_0 = grad(H0, rho)[0]

        def f_(z_):
            H = self._hamiltonian(recon_x, x, z_, rho, G_inv, G_log_det)
            grho = grad(H, rho, retain_graph=True)[0]
            return z + 0.5 * self.eps_lf * (grho_0 + grho)

        z_ = z.clone()
        for _ in range(steps):
            z_ = f_(z_)
        return z_

    def _leap_step_3(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves third equation of generalized leapfrog integrator
        """
        H = self._hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        gz = grad(H, z, create_graph=True)[0]
        return rho - 0.5 * self.eps_lf * gz

    def _hamiltonian(self, recon_x, x, z, rho, G_inv=None, G_log_det=None):
        """
        Computes the Hamiltonian function.
        used for HVAE and RHVAE
        """
        norm = (
            torch.transpose(rho.unsqueeze(-1), 1, 2) @ G_inv @ rho.unsqueeze(-1)
        ).sum()

        return -self._log_p_xz(recon_x, x, z).sum() + 0.5 * norm + 0.5 * G_log_det.sum()

    def update_metric(self):
        r"""
        As soon as the model has seen all the data points (i.e. at the end of 1 loop)
        we update the final metric function using \mu(x_i) as centroids
        """
        # convert to 1 big tensor
        self.M_tens = torch.cat(self.M)
        self.centroids_tens = torch.cat(self.centroids)

        # define new metric
        def G(z):
            return torch.inverse(
                (
                    self.M_tens.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(
                            self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                        )
                        ** 2
                        / (self.temperature ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1)
                + self.lbd * torch.eye(self.latent_dim).to(z.device)
            )

        def G_inv(z):
            return (
                self.M_tens.unsqueeze(0)
                * torch.exp(
                    -torch.norm(
                        self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (self.temperature ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(z.device)

        self.G = G
        self.G_inv = G_inv
        self.M = []
        self.centroids = []

    def loss_function(
        self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var, G_inv, G_log_det
    ):

        logpxz = self._log_p_xz(recon_x, x, zK)  # log p(x, z_K)
        logrhoK = (
            (
                -0.5
                * (
                    torch.transpose(rhoK.unsqueeze(-1), 1, 2)
                    @ G_inv
                    @ rhoK.unsqueeze(-1)
                )
                .squeeze()
                .squeeze()
                - 0.5 * G_log_det
            )
            - torch.log(torch.tensor([2 * np.pi]).to(x.device)) * self.latent_dim / 2
        )  # log p(\rho_K)

        logp = logpxz + logrhoK

        # define a N(0, I) distribution
        normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim).to(x.device),
            covariance_matrix=torch.eye(self.latent_dim).to(x.device),
        )

        logq = normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # log(q(z_0|x))

        return -(logp - logq).mean()

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _tempering(self, k, K):
        """Perform tempering step"""

        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k

    def _log_p_x_given_z(self, recon_x, x, reduction="none"):
        r"""Estimate the decoder's log-density modelled as follows:
            p(x|z)     = \prod_i Bernouilli(x_i|pi_{theta}(z_i))
            p(x = s|z) = \prod_i (pi(z_i))^x_i * (1 - pi(z_i)^(1 - x_i))"""
        return -F.binary_cross_entropy(
            recon_x, x.reshape(-1, self.input_dim), reduction=reduction
        ).sum(dim=1)

    def _log_z(self, z):
        """
        Return Normal density function as prior on z
        """

        # define a N(0, I) distribution
        normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim).to(z.device),
            covariance_matrix=torch.eye(self.latent_dim).to(z.device),
        )
        return normal.log_prob(z)

    def _log_p_xz(self, recon_x, x, z):
        """
        Estimate log(p(x, z)) using Bayes rule
        """
        logpxz = self._log_p_x_given_z(recon_x, x)
        logpz = self._log_z(z)
        return logpxz + logpz

    def likelihood(self, x, sample_size=10):
        r"""
        Estimate the likelihood of the model :math:`\log(p(x))` using importance sampling on
        :math:`q_{\phi}(z|x)`
        """
        # print(sample_size)
        mu, log_var = self.encode(x.reshape(-1, self.input_dim))
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=x.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)

        Z0 = Z

        recon_X = self.decode(Z)

        # get metric value
        G_rep = self.G(Z)
        G_inv_rep = self.G_inv(Z)

        G_log_det_rep = torch.logdet(G_rep)

        L_rep = torch.linalg.cholesky(G_rep)

        G_inv_rep_0 = G_inv_rep
        G_log_det_rep_0 = G_log_det_rep

        # initialization
        gamma = torch.randn_like(Z0, device=x.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        rho = (L_rep @ rho.unsqueeze(-1)).squeeze(
            -1
        )  # sample from the multivariate N(0, G)

        rho0 = rho

        X_rep = x.repeat(sample_size, 1, 1, 1).reshape(-1, self.input_dim)

        for k in range(self.n_lf):

            # step 1
            rho_ = self._leap_step_1(recon_X, X_rep, Z, rho, G_inv_rep, G_log_det_rep)

            # step 2
            Z = self._leap_step_2(recon_X, X_rep, Z, rho_, G_inv_rep, G_log_det_rep)

            recon_X = self.decode(Z)

            G_rep_inv = self.G_inv(Z)
            G_log_det_rep = -torch.logdet(G_rep_inv)

            # step 3
            rho__ = self._leap_step_3(recon_X, X_rep, Z, rho_, G_inv_rep, G_log_det_rep)

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        bce = F.binary_cross_entropy(recon_X, X_rep, reduction="none")

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, self.input_dim).sum(dim=2)  # log(p(X|Z))

        logpz = self._log_z(Z).reshape(sample_size, -1)  # log(p(Z))

        logrho0 = (
            (
                -0.5
                * (
                    torch.transpose(rho0.unsqueeze(-1), 1, 2)
                    * self.beta_zero_sqrt
                    @ G_inv_rep_0
                    @ rho0.unsqueeze(-1)
                    * self.beta_zero_sqrt
                )
                .squeeze()
                .squeeze()
                - 0.5 * G_log_det_rep_0
            )
            - torch.log(torch.tensor([2 * np.pi]).to(x.device)) * self.latent_dim / 2
        ).reshape(sample_size, -1)

        # log(p(\rho_0))
        logrho = (
            (
                -0.5
                * (
                    torch.transpose(rho.unsqueeze(-1), 1, 2)
                    @ G_inv_rep
                    @ rho.unsqueeze(-1)
                )
                .squeeze()
                .squeeze()
                - 0.5 * G_log_det_rep
            )
            - torch.log(torch.tensor([2 * np.pi]).to(x.device)) * self.latent_dim / 2
        ).reshape(sample_size, -1)
        # log(p(\rho_K))

        # define a N(0, I) distribution
        normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim).to(x.device),
            covariance_matrix=torch.eye(self.latent_dim).to(x.device),
        )

        logqzx = normal.log_prob(Eps) - 0.5 * log_var.sum(dim=1)  # log(q(Z_0|X))

        logpx = (logpxz + logpz + logrho - logrho0 - logqzx).logsumexp(dim=0).mean(
            dim=0
        ) - torch.log(
            torch.Tensor([sample_size]).to(x.device)
        )  # + self.latent_dim /2 * torch.log(self.beta_zero_sqrt ** 2)

        return logpx
