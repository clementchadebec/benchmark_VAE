import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from ...models import VAE
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOuput
from ..nn import BaseEncoder, BaseDecoder
from .hvae_config import HVAEConfig

from typing import Optional

class HVAE(VAE):
    r"""
    This is an implementation of the Hamiltonian VAE models proposed in 
    (https://proceedings.neurips.cc/paper/2018/file/3202111cf90e7c816a472aaceb72b0df-Paper.pdf). 
    This models combines Hamiltonian and normalizing flows together to improve the true posterior 
    estimate within the VAE framework.

    Args:
        model_config (HVAEConfig): A model configuration setting the main parameters of the model

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
        model_config: HVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(
                self, model_config=model_config, encoder=encoder, decoder=decoder
            )

        self.model_name = "HVAE"

        self.n_lf = model_config.n_lf
        self.eps_lf =  nn.Parameter(
            torch.tensor([model_config.eps_lf]),
            requires_grad=True if model_config.learn_eps_lf else False)
        self.beta_zero_sqrt =  nn.Parameter(
            torch.tensor([model_config.beta_zero])**0.5,
            requires_grad=True if model_config.learn_beta_zero else False)


    def forward(self, inputs: BaseDataset) -> ModelOuput:
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

        z0, eps0 = self._sample_gauss(mu, log_var)
        gamma = torch.randn_like(z0, device=x.device)
        rho = gamma / self.beta_zero_sqrt

        z = z0
        beta_sqrt_old = self.beta_zero_sqrt

        recon_x = self.decoder(z)['reconstruction']

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

            recon_x = self.decoder(z)['reconstruction']

            U = -self._log_p_xz(recon_x, x, z).sum()
            g = grad(U, z, create_graph=True)[0]

            # 3rd leapfrog step
            rho__ = rho_ - (self.eps_lf / 2) * g

            # tempering steps
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt


        loss = self.loss_function(recon_x, x, z0, z, rho, eps0, gamma, mu, log_var)

        output = ModelOuput(
            loss=loss,
            recon_x=recon_x,
            z=z,
            z0=z0,
            rho=rho,
            eps0=eps0,
            gamma=gamma,
            mu=mu,
            log_var=log_var
        )

        return output

    def loss_function(self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var):

        normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.model_config.latent_dim).to(x.device),
            covariance_matrix=torch.eye(self.model_config.latent_dim).to(x.device),
        )

        logpxz = self._log_p_xz(recon_x.reshape(x.shape[0], -1), x, zK)  # log p(x, z_K)
        logrhoK = normal.log_prob(rhoK)  # log p(\rho_K)
        logp = logpxz + logrhoK

        logq = normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # q(z_0|x)

        return -(logp - logq).sum()


    def _tempering(self, k, K):
        """Perform tempering step"""

        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k

    def _log_p_x_given_z(self, recon_x, x):

        if self.model_config.reconstruction_loss == 'mse':

            recon_loss =  -F.mse_loss(
                recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction='none'
            ).sum()

        elif self.model_config.reconstruction_loss == 'bce':

            recon_loss = F.binary_cross_entropy(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
            ).sum()
        
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


    def _hamiltonian(self, recon_x, x, z, rho, G_inv=None, G_log_det=None):
        """
        Computes the Hamiltonian function.
        used for HVAE
        """
        return -self.log_p_xz(recon_x.reshape(x.shape[0], -1), x, z).sum()



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
                a ``model_config.json`` and a ``model.pt`` if no custom architectures were
                provided

                or
                a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
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

