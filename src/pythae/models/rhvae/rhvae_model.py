import inspect
import logging
import os
import warnings
from collections import deque
from copy import deepcopy
from typing import Optional

import cloudpickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from ...customexception import BadInheritanceError
from ...data.datasets import BaseDataset
from ..base.base_utils import CPU_Unpickler, ModelOutput, hf_hub_is_available
from ..nn import BaseDecoder, BaseEncoder, BaseMetric
from ..nn.default_architectures import Metric_MLP
from ..vae import VAE
from .rhvae_config import RHVAEConfig
from .rhvae_utils import create_inverse_metric, create_metric

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class RHVAE(VAE):
    r"""
    Riemannian Hamiltonian VAE model.


    Args:
        model_config (RHVAEConfig): A model configuration setting the main parameters of the model.

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
        model_config: RHVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        metric: Optional[BaseMetric] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "RHVAE"

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
        self.M = deque(maxlen=100)
        self.centroids = deque(maxlen=100)

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
        r"""
        As soon as the model has seen all the data points (i.e. at the end of 1 loop)
        we update the final metric function using \mu(x_i) as centroids
        """
        self._update_metric()

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
                    "pythae.models.base_architectures.BaseMetric. Refer to documentation."
                )
            )

        self.metric = metric

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        r"""
        The input data is first encoded. The reparametrization is used to produce a sample
        :math:`z_0` from the approximate posterior :math:`q_{\phi}(z|x)`. Then Riemannian
        Hamiltonian equations are solved using the generalized leapfrog integrator. In the meantime,
        the input data :math:`x` is fed to the metric network outputing the matrices
        :math:`L_{\psi}`. The metric is computed and used with the integrator.

        Args:
            inputs (BaseDataset): The training data with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]

        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z0, eps0 = self._sample_gauss(mu, std)

        z = z0

        if self.training:
            # update the metric using batch data points
            L = self.metric(x)["L"]

            M = L @ torch.transpose(L, 1, 2)

            # store LL^T and mu(x_i) to update final metric
            self.M.append(M.detach().clone())
            self.centroids.append(mu.detach().clone())

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

        recon_x = self.decoder(z)["reconstruction"]

        for k in range(self.n_lf):

            # perform leapfrog steps

            # step 1
            rho_ = self._leap_step_1(recon_x, x, z, rho, G_inv, G_log_det)

            # step 2
            z = self._leap_step_2(recon_x, x, z, rho_, G_inv, G_log_det)

            recon_x = self.decoder(z)["reconstruction"]

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
            G_inv=G_inv,
            G_log_det=G_log_det,
        )

        return output

    def predict(self, inputs: torch.Tensor) -> ModelOutput:
        """The input data is encoded and decoded without computing loss

        Args:
            inputs (torch.Tensor): The input data to be reconstructed, as well as to generate the embedding.

        Returns:
            ModelOutput: An instance of ModelOutput containing reconstruction, raw embedding (output of encoder), and the final embedding (output of metric)
        """
        encoder_output = self.encoder(inputs)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z0, _ = self._sample_gauss(mu, std)

        z = z0

        G = self.G(z)
        G_inv = self.G_inv(z)
        L = torch.linalg.cholesky(G)

        G_log_det = -torch.logdet(G_inv)

        gamma = torch.randn_like(z0, device=inputs.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        # sample \rho from N(0, G)
        rho = (L @ rho.unsqueeze(-1)).squeeze(-1)

        recon_x = self.decoder(z)["reconstruction"]

        for k in range(self.n_lf):

            # perform leapfrog steps

            # step 1
            rho_ = self._leap_step_1(recon_x, inputs, z, rho, G_inv, G_log_det)

            # step 2
            z = self._leap_step_2(recon_x, inputs, z, rho_, G_inv, G_log_det)

            recon_x = self.decoder(z)["reconstruction"]

            # compute metric value on new z using final metric
            G = self.G(z)
            G_inv = self.G_inv(z)

            G_log_det = -torch.logdet(G_inv)

            # step 3
            rho__ = self._leap_step_3(recon_x, inputs, z, rho_, G_inv, G_log_det)

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        output = ModelOutput(
            recon_x=recon_x,
            raw_embedding=encoder_output.embedding,
            embedding=z if self.n_lf > 0 else encoder_output.embedding,
        )

        return output

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
        used for RHVAE
        """
        norm = (
            torch.transpose(rho.unsqueeze(-1), 1, 2) @ G_inv @ rho.unsqueeze(-1)
        ).sum()

        return -self._log_p_xz(recon_x, x, z).sum() + 0.5 * norm + 0.5 * G_log_det.sum()

    def _update_metric(self):
        # convert to 1 big tensor

        self.M_tens = torch.cat(list(self.M))
        self.centroids_tens = torch.cat(list(self.centroids))

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
        self.M = deque(maxlen=100)
        self.centroids = deque(maxlen=100)

    def loss_function(
        self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var, G_inv, G_log_det
    ):

        logpxz = self._log_p_xz(recon_x, x, zK)  # log p(x, z_K)
        logrhoK = (
            -0.5
            * (torch.transpose(rhoK.unsqueeze(-1), 1, 2) @ G_inv @ rhoK.unsqueeze(-1))
            .squeeze()
            .squeeze()
            - 0.5 * G_log_det
            # - torch.log(torch.tensor([2 * np.pi]).to(x.device)) * self.latent_dim / 2
        )  # log p(\rho_K)

        logp = logpxz + logrhoK

        # define a N(0, I) distribution
        normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim).to(x.device),
            covariance_matrix=torch.eye(self.latent_dim).to(x.device),
        )

        logq = normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # log(q(z_0|x))

        return -(logp - logq).mean(dim=0)

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

    def _log_p_x_given_z(self, recon_x, x):

        if self.model_config.reconstruction_loss == "mse":
            # sigma is taken as I_D
            recon_loss = (
                -0.5
                * F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1)
            )
            -torch.log(torch.tensor([2 * np.pi]).to(x.device)) * np.prod(
                self.input_dim
            ) / 2

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
        return normal.log_prob(z)

    def _log_p_xz(self, recon_x, x, z):
        """
        Estimate log(p(x, z)) using Bayes rule
        """
        logpxz = self._log_p_x_given_z(recon_x, x)
        logpz = self._log_z(z)
        return logpxz + logpz

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
                z0, eps = self._sample_gauss(mu, std)
                gamma = torch.randn_like(z0, device=x.device)
                rho = gamma / self.beta_zero_sqrt

                z = z0
                beta_sqrt_old = self.beta_zero_sqrt

                G = self.G(z0)
                G_inv = self.G_inv(z0)
                G_log_det = -torch.logdet(G_inv)
                L = torch.linalg.cholesky(G)

                # initialization
                gamma = torch.randn_like(z0, device=z.device)
                rho = gamma / self.beta_zero_sqrt
                beta_sqrt_old = self.beta_zero_sqrt

                rho = (L @ rho.unsqueeze(-1)).squeeze(
                    -1
                )  # sample from the multivariate N(0, G)

                recon_x = self.decoder(z)["reconstruction"]

                for k in range(self.n_lf):

                    # perform leapfrog steps

                    # step 1
                    rho_ = self._leap_step_1(recon_x, x_rep, z, rho, G_inv, G_log_det)

                    # step 2
                    z = self._leap_step_2(recon_x, x_rep, z, rho_, G_inv, G_log_det)

                    recon_x = self.decoder(z)["reconstruction"]

                    G_inv = self.G_inv(z)
                    G_log_det = -torch.logdet(G_inv)

                    # step 3
                    rho__ = self._leap_step_3(recon_x, x_rep, z, rho_, G_inv, G_log_det)

                    # tempering steps
                    beta_sqrt = self._tempering(k + 1, self.n_lf)
                    rho = (beta_sqrt_old / beta_sqrt) * rho__
                    beta_sqrt_old = beta_sqrt

                log_q_z0_given_x = -0.5 * (
                    log_var + (z0 - mu) ** 2 / torch.exp(log_var)
                ).sum(dim=-1)
                log_p_z = -0.5 * (z ** 2).sum(dim=-1)

                log_p_rho0 = normal.log_prob(gamma) - torch.logdet(
                    L / self.beta_zero_sqrt
                )  # rho0 ~ N(0, 1/beta_0 * G(z0))

                log_p_rho = (
                    -0.5
                    * (
                        torch.transpose(rho.unsqueeze(-1), 1, 2)
                        @ G_inv
                        @ rho.unsqueeze(-1)
                    )
                    .squeeze()
                    .squeeze()
                    - 0.5 * G_log_det
                ) - torch.log(
                    torch.tensor([2 * np.pi]).to(z.device)
                ) * self.latent_dim / 2  # rho0 ~ N(0, G(z))

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
                    log_p_x_given_z
                    + log_p_z
                    + log_p_rho
                    - log_p_rho0
                    - log_q_z0_given_x
                )  # N*log(2*pi) simplifies in prior and posterior

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())

        return np.mean(log_p)

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
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.metric))
                cloudpickle.dump(self.metric, fp)

        torch.save(model_dict, os.path.join(model_path, "model.pt"))

    @classmethod
    def _load_custom_metric_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)

        if "metric.pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing metric pkl file ('metric.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom metrics."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "metric.pkl"), "rb") as fp:
                metric = CPU_Unpickler(fp).load()

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

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
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

    @classmethod
    def load_from_hf_hub(
        cls, hf_hub_path: str, allow_pickle: bool = False
    ):  # pragma: no cover
        """Class method to be used to load a pretrained model from the Hugging Face hub

        Args:
            hf_hub_path (str): The path where the model should have been be saved on the
                hugginface hub.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl`` and ``metric.pkl``) if a custom encoder (resp. decoder and/or
                metric) was provided
        """

        if not hf_hub_is_available():
            raise ModuleNotFoundError(
                "`huggingface_hub` package must be installed to load models from the HF hub. "
                "Run `python -m pip install huggingface_hub` and log in to your account with "
                "`huggingface-cli login`."
            )

        else:
            from huggingface_hub import hf_hub_download

        logger.info(f"Downloading {cls.__name__} files for rebuilding...")

        config_path = hf_hub_download(repo_id=hf_hub_path, filename="model_config.json")
        dir_path = os.path.dirname(config_path)

        _ = hf_hub_download(repo_id=hf_hub_path, filename="model.pt")

        model_config = cls._load_model_config_from_folder(dir_path)

        if (
            cls.__name__ + "Config" != model_config.name
            and cls.__name__ + "_Config" != model_config.name
        ):
            warnings.warn(
                f"You are trying to load a "
                f"`{ cls.__name__}` while a "
                f"`{model_config.name}` is given."
            )

        model_weights = cls._load_model_weights_from_folder(dir_path)

        if (
            not model_config.uses_default_encoder
            or not model_config.uses_default_decoder
            or not model_config.uses_default_metric
        ) and not allow_pickle:
            warnings.warn(
                "You are about to download pickled files from the HF hub that may have "
                "been created by a third party and so could potentially harm your computer. If you "
                "are sure that you want to download them set `allow_pickle=true`."
            )

        else:

            if not model_config.uses_default_encoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename="encoder.pkl")
                encoder = cls._load_custom_encoder_from_folder(dir_path)

            else:
                encoder = None

            if not model_config.uses_default_decoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename="decoder.pkl")
                decoder = cls._load_custom_decoder_from_folder(dir_path)

            else:
                decoder = None

            if not model_config.uses_default_metric:
                _ = hf_hub_download(repo_id=hf_hub_path, filename="metric.pkl")
                metric = cls._load_custom_metric_from_folder(dir_path)

            else:
                metric = None

            logger.info(f"Successfully downloaded {cls.__name__} model!")

            model = cls(model_config, encoder=encoder, decoder=decoder, metric=metric)

            metric_M, metric_centroids = cls._load_metric_matrices_and_centroids(
                dir_path
            )

            model.M_tens = metric_M
            model.centroids_tens = metric_centroids

            model.G = create_metric(model)
            model.G_inv = create_inverse_metric(model)

            model.load_state_dict(model_weights)

            return model
