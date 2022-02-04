import torch
import os

from ...models import VAE
from .iwae_config import IWAEConfig
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseEncoder, BaseDecoder
from ..nn.default_architectures import Encoder_VAE_MLP

from typing import Optional

import torch.nn.functional as F


class IWAE(VAE):
    """
    Importance Weighted Autoencoder model.
    
    Args:
        model_config(IWAEConfig): The IWAE configuration seting the main 
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
        model_config: IWAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "IWAE"
        self.beta = model_config.beta
        self.n_samples = model_config.number_samples

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training datasat with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        mu = mu.unsqueeze(1).repeat(1, self.n_samples, 1)
        log_var = log_var.unsqueeze(1).repeat(1, self.n_samples, 1)

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z.reshape(-1, self.latent_dim))["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)

        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = F.mse_loss(
                recon_x.reshape(recon_x.shape[0], -1),
                x.reshape(x.shape[0], -1).unsqueeze(1).repeat(
                    1, self.n_samples, 1).reshape(recon_x.shape[0], -1),
                reduction='none'
            ).sum(dim=-1).reshape(x.shape[0], -1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(recon_x.shape[0], -1),
                x.reshape(x.shape[0], -1).unsqueeze(1).repeat(
                    1, self.n_samples, 1).reshape(recon_x.shape[0], -1),
                reduction='none'
            ).sum(dim=-1).reshape(x.shape[0], -1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        log_w = recon_loss + self.beta * KLD

        w_tilde = F.softmax(log_w, dim=-1)

        return (
            (w_tilde * log_w).sum(dim=-1).mean(dim=0),
            recon_loss.mean(),
            KLD.mean()
        )

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
        model_config = IWAEConfig.from_json_file(path_to_model_config)

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
