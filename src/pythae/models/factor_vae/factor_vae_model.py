import os
from typing import Optional

import torch
import torch.nn.functional as F

from ...customexception import BadInheritanceError
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseDiscriminator, BaseEncoder
from ..vae import VAE
from .factor_vae_config import FactorVAEConfig
from .factor_vae_utils import FactorVAEDiscriminator


class FactorVAE(VAE):
    """
    FactorVAE model.

    Args:
        model_config (FactorVAEConfig): The Variational Autoencoder configuration setting the main
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
        model_config: FactorVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.discriminator = FactorVAEDiscriminator(latent_dim=model_config.latent_dim)

        self.model_name = "FactorVAE"
        self.gamma = model_config.gamma

    def set_discriminator(self, discriminator: BaseDiscriminator) -> None:
        r"""This method is called to set the discriminator network

        Args:
            discriminator (BaseDiscriminator): The discriminator module that needs to be set to the model.

        """
        if not issubclass(type(discriminator), BaseDiscriminator):
            raise BadInheritanceError(
                (
                    "Discriminator must inherit from BaseDiscriminator class from "
                    "pythae.models.base_architectures.BaseDiscriminator. Refer to documentation."
                )
            )

        self.discriminator = discriminator

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        # first batch
        x = inputs["data"]

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        # second batch
        x_bis = inputs["data_bis"]

        encoder_output = self.encoder(x_bis)

        mu_bis, log_var_bis = encoder_output.embedding, encoder_output.log_covariance

        std_bis = torch.exp(0.5 * log_var_bis)
        z_bis, _ = self._sample_gauss(mu_bis, std_bis)

        z_bis_permuted = self._permute_dims(z_bis).detach()

        recon_loss, autoencoder_loss, discriminator_loss = self.loss_function(
            recon_x, x, mu, log_var, z, z_bis_permuted
        )

        loss = autoencoder_loss + discriminator_loss

        output = ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            autoencoder_loss=autoencoder_loss,
            discriminator_loss=discriminator_loss,
            recon_x=recon_x,
            z=z,
            z_bis_permuted=z_bis_permuted,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z, z_bis_permuted):

        N = z.shape[0]  # batch size

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        latent_adversarial_score = self.discriminator(z)

        TC = (latent_adversarial_score[:, 0] - latent_adversarial_score[:, 1]).mean()
        autoencoder_loss = recon_loss + KLD + self.gamma * TC

        # discriminator loss
        permuted_latent_adversarial_score = self.discriminator(z_bis_permuted)

        true_labels = (
            torch.ones(N, requires_grad=False).type(torch.LongTensor).to(z.device)
        )
        fake_labels = (
            torch.zeros(N, requires_grad=False).type(torch.LongTensor).to(z.device)
        )

        TC_permuted = F.cross_entropy(
            latent_adversarial_score, fake_labels
        ) + F.cross_entropy(permuted_latent_adversarial_score, true_labels)

        discriminator_loss = 0.5 * TC_permuted

        return (
            (recon_loss).mean(dim=0),
            (autoencoder_loss).mean(dim=0),
            (discriminator_loss).mean(dim=0),
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _permute_dims(self, z):
        permuted = torch.zeros_like(z)

        for i in range(z.shape[-1]):
            perms = torch.randperm(z.shape[0]).to(z.device)
            permuted[:, i] = z[perms, i]

        return permuted

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = FactorVAEConfig.from_json_file(path_to_model_config)

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
