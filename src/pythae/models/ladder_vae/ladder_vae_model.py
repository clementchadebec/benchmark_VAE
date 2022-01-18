import torch
import os

from ...models import VAE
from .ladder_vae_config import LadderVAEConfig
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseEncoder, BaseDecoder
from ..nn.default_architectures import Encoder_LadderVAE_MLP

from typing import Optional

import torch.nn.functional as F


class LadderVAE(VAE):
    r"""
    Ladder VAE model.
    
    Args:
        model_config(LadderVAEConfig): The Variational Autoencoder configuration seting the main 
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
        model_config: LadderVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "LadderVAE"
        self.latent_dimensions = latent_dimensions
        self.beta = model_config.beta

        assert len(self.latent_dimensions) == self.encoder.depth - 1 , (
            'Please ensure that the number of latent dimensions provided fits the depth of the '\
            f'encoder. Got {len(self.latent_dimensions)} latent dimension and encoder of depth '\
            f'{self.encoder.depth}.'
        )

    def forward(self, inputs: BaseDataset):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training datasat with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        output_layers_levels = torch.arange(1, len(self.latent_dimensions))

        encoder_output = self.encoder(x, output_layer_levels=output_layer_levels)

        mu_p = []
        log_var_p = []

        for i in range(len(self.latent_dimensions)-1):
            mu_p.append(encoder_output[f"embedding_layer_{i+1}"])
            log_var_p.append(encoder_output[f"log_covariance_layer_{i+1}"])

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        decoder_output = self.decoder(z)

        
        mu_q = []
        log_var_q = []
        z_q = []

        mu_new = []
        log_var_new = []

        for i in range(len(self.latent_dimensions)-1)[::-1]:
            mu_q.append(decoder_output[f"embedding_layer_{i+1}"])
            log_var_q.append(decoder_output[f"log_covariance_layer_{i+1}"])
            z_q.append(decoder_output[f"z_layer_{i+1}"])

            mu_updated, log_var_updated = self._update_mu_log_var(
                mu_p[i], log_var_p[i], mu_q[i], log_var_q[i])

            mu_new.append()

        loss, recon_loss, kld = self.loss_function(
            recon_x,
            x,
            mu,
            log_var,
            z,
            mu_p,
            log_var_p,
            z_q,
            mu_q,
            log_var_q
        )

        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z, mu_p, log_var_p, z_q, mu_q, log_var_q):

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


        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return (recon_loss + self.beta * KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _log_gaussian_density(self, z, mu, log_var):
        return (
            -0.5 * (log_var + (z - mu) ** 2 / log_var.exp())
        ).sum(dim=-1)

    def _log_normal_density(self, z):
        return -(0.5 * z ** 2).sum(dim=-1)

    def _update_mu_log_var(self, mu_p, log_var_p, mu_q, log_var_q):
        
        mu_new = (
            mu_p / log_var_p.exp() + mu_q / log_var_q.exp()) / (
                1 / log_var_p.exp() + 1 / log_var_q.exp()
            )

        log_var_new = (1 / (1 / log_var_p.exp() + 1 / log_var_q.exp())).log()

        return (mu_new, log_var_new)

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
        model_config = BetaVAEConfig.from_json_file(path_to_model_config)

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
