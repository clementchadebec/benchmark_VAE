import inspect
import logging
import os
import warnings
from copy import deepcopy
from typing import Optional

import cloudpickle
import torch
import torch.nn.functional as F

from ...customexception import BadInheritanceError
from ...data.datasets import BaseDataset
from ..base.base_utils import CPU_Unpickler, ModelOutput, hf_hub_is_available
from ..nn import BaseDecoder, BaseDiscriminator, BaseEncoder
from ..nn.default_architectures import Discriminator_MLP
from ..vae import VAE
from .adversarial_ae_config import Adversarial_AE_Config

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class Adversarial_AE(VAE):
    """Adversarial Autoencoder model.

    Args:
        model_config (Adversarial_AE_Config): The Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        discriminator (BaseDiscriminator): An instance of BaseDiscriminator (inheriting from
            `torch.nn.Module` which plays the role of discriminator. This argument allows you to
            use your own neural networks architectures if desired. If None is provided, a simple
            Multi Layer Preception (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used.
            Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: Adversarial_AE_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        discriminator: Optional[BaseDiscriminator] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        if discriminator is None:
            if model_config.latent_dim is None:
                raise AttributeError(
                    "No latent dimension provided !"
                    "'latent_dim' parameter of Adversarial_AE_Config instance "
                    "must be set to a value. Unable to build discriminator automatically."
                )

            self.model_config.discriminator_input_dim = self.model_config.latent_dim

            discriminator = Discriminator_MLP(model_config)
            self.model_config.uses_default_discriminator = True

        else:
            self.model_config.uses_default_discriminator = False

        self.set_discriminator(discriminator)

        self.model_name = "Adversarial_AE"

        assert (
            0 <= self.model_config.adversarial_loss_scale <= 1
        ), "adversarial_loss_scale must be in [0, 1]"

        self.adversarial_loss_scale = self.model_config.adversarial_loss_scale
        self.reconstruction_loss_scale = self.model_config.reconstruction_loss_scale
        self.deterministic_posterior = (
            1 if self.model_config.deterministic_posterior else 0
        )

    def set_discriminator(self, discriminator: BaseDiscriminator) -> None:
        r"""This method is called to set the discriminator network

        Args:
            discriminator (BaseDiscriminator): The discriminator module that needs to be set to the
                model.

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
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = (1 - self.deterministic_posterior) * torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        z_prior = torch.randn_like(z, device=x.device).requires_grad_(True)

        recon_loss, autoencoder_loss, discriminator_loss = self.loss_function(
            recon_x, x, z, z_prior
        )

        loss = autoencoder_loss + discriminator_loss

        output = ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            autoencoder_loss=autoencoder_loss,
            discriminator_loss=discriminator_loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, z, z_prior):

        N = z.shape[0]  # batch size

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

        gen_adversarial_score = self.discriminator(z).embedding.flatten()
        prior_adversarial_score = self.discriminator(z_prior).embedding.flatten()

        true_labels = torch.ones(N, requires_grad=False).to(z.device)
        fake_labels = torch.zeros(N, requires_grad=False).to(z.device)

        autoencoder_loss = self.adversarial_loss_scale * (
            F.binary_cross_entropy(
                gen_adversarial_score, true_labels
            )  # generated are true
        ) + (recon_loss * self.reconstruction_loss_scale)

        gen_adversarial_score_ = self.discriminator(z.detach()).embedding.flatten()

        discriminator_loss = (
            F.binary_cross_entropy(
                prior_adversarial_score, true_labels
            )  # prior is true
        ) + (
            F.binary_cross_entropy(
                gen_adversarial_score_, fake_labels
            )  # generated are false
        )

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

    def save(self, dir_path: str):
        """Method to save the model at a specific location

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """

        # This creates the dir if not available
        super().save(dir_path)
        model_path = dir_path

        model_dict = {"model_state_dict": deepcopy(self.state_dict())}

        if not self.model_config.uses_default_discriminator:
            with open(os.path.join(model_path, "discriminator.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(
                    inspect.getmodule(self.discriminator)
                )
                cloudpickle.dump(self.discriminator, fp)

        torch.save(model_dict, os.path.join(model_path, "model.pt"))

    @classmethod
    def _load_custom_discriminator_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)

        if "discriminator.pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing discriminator pkl file ('discriminator.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom discriminators."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "discriminator.pkl"), "rb") as fp:
                discriminator = CPU_Unpickler(fp).load()

        return discriminator

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

        if not model_config.uses_default_discriminator:
            discriminator = cls._load_custom_discriminator_from_folder(dir_path)

        else:
            discriminator = None

        model = cls(
            model_config, encoder=encoder, decoder=decoder, discriminator=discriminator
        )
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
                ``decoder.pkl`` and ``discriminator``) if a custom encoder (resp. decoder and/or
                discriminator) was provided
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
            or not model_config.uses_default_discriminator
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

            if not model_config.uses_default_discriminator:
                _ = hf_hub_download(repo_id=hf_hub_path, filename="discriminator.pkl")
                discriminator = cls._load_custom_discriminator_from_folder(dir_path)

            else:
                discriminator = None

            logger.info(f"Successfully downloaded {cls.__name__} model!")

            model = cls(
                model_config,
                encoder=encoder,
                decoder=decoder,
                discriminator=discriminator,
            )
            model.load_state_dict(model_weights)

            return model
