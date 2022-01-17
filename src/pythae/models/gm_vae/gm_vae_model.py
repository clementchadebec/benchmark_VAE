import torch
import torch.nn.functional as F
import os
import numpy as np

from ...models import VAE
from .gm_vae_config import GMVAEConfig
from .gm_vae_utils import MixtureGenerator
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseEncoder, BaseDecoder
from ..nn.default_architectures import Encoder_VAE_MLP

from typing import Optional

import torch.nn.functional as F


class GMVAE(VAE):
    r"""
    Gaussian Mixture VAE model.
    
    Args:
        model_config(GMVAEConfig): The Variational Autoencoder configuration seting the main 
        parameters of the model

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
        model_config: GMVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "GMVAE"
    
        mixture_generator = MixtureGenerator(model_config)
        self.mixture_generator = mixture_generator

    def set_mixture_encoder(self, mixture_encoder):
        r"""This method is called to set the mixture encoder network outputing the
        :logits values for cluster sampling. 

        Args:
            mixture_encoder (BaseEncoder): The encoder module that needs to be set to the model.

        """
        if not issubclass(type(mixture_encoder), BaseEncoder):
            raise BadInheritanceError(
                (
                    "Mixtrue encoder must inherit from BaseEncoder class from "
                    "pythae.models.base_architectures.BaseEncoder. Refer to documentation."
                )
            )

        self.mixture_encoder = mixture_encoder

    def forward(self, inputs: BaseDataset):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training datasat with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        encoder_output = self.encoder(x)

        mu_z, log_var_z = encoder_output.embedding_z, encoder_output.log_covariance_z # GMM
        mu_w, log_var_w = encoder_output.embedding_w, encoder_output.log_covariance_w # N(0, I)

        std_z = torch.exp(0.5 * log_var_z)
        std_w = torch.exp(0.5 * log_var_w)
        z, eps_z = self._sample_gauss(mu_z, std_z)
        w, eps_w = self._sample_gauss(mu_w, std_w)

        mixture_generator_output = self.mixture_generator(mu) # Provides the parameters of the mixture
        
        gmm_means, gmm_log_var = (
            mixture_generator_output.gmm_means,
            mixture_generator_output.gmm_log_covariances
        )

        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(
            recon_x,
            x,
            mu_z,
            log_var_z,
            z,
            mu_w,
            log_var_w,
            w,
            gmm_means,
            gmm_log_var)

        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(
        self,
        recon_x,
        x,
        mu_z,
        log_var_z,
        z,
        mu_w,
        log_var_w,
        w,
        gmm_means,
        gmm_log_var
    ):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction='none'
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction='none'
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var_w - mu_w.pow(2) - log_var_w.exp(), dim=-1) # w ~ N(0, I)

        KLD_gmm = self.compute_KLD_gmm(
            mu_z,
            log_var_z,
            gmm_means,gmm_log_var
        )

        p_y_given_zw = compute_log_p_y_given_zw(z, gmm_means, gmm_log_var)

        KLD_categorical = (p_y_given_zw.log() * p_y_given_zw - \
            torch.log(1 / self.model_config.number_components)).sum(dim=-1)

        return (
            (recon_loss + KLD + KLD_gmm + KLD_categorical).mean(dim=0),
            recon_loss.mean(dim=0),
            KLD.mean(dim=0),
            KLD_gmm.mean(dim=0),
            KLD_categorical.mean(dim=0)
        )

    def compute_KLD_gmm(
        self,
        mu_z,
        log_var_z,
        gmm_means,
        gmm_log_var
        ):

            mu_z = mu_z.unsqueeze(-1)
            log_var_z = log_var_z.unsqueeze(-1)

            KLD_gmm = 0.5 * (
                gmm_log_var - log_var_z + \
                (gmm_means - mu_z) ** 2 / gmm_log_var.exp() + \
                log_var_z.exp() / gmm_log_var.exp()
            )

            return KLD_gmm.sum(dim=-1)

    def compute_p_y_given_zw(self, z, gmm_means, gmm_log_var):
        
        z = z.unsqueeze(-1)
        
        log_p_z = -0.5 * (
            (z - gmm_means) ** 2 / gmm_log_var - \
            gmm_log_var).sum(dim=1) - \
            z.shape[1] / 2 * torch.log(torch.tensor([2 * np.pi]).to(z.device))

        return F.softmax(log_p_z)


    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = GMVAEConfig.from_json_file(path_to_model_config)

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
                
            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
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
