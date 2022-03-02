import torch
import os
import dill

from copy import deepcopy
from ...customexception import BadInheritanceError
from ...models import VAE
from .vae_gan_config import VAEGANConfig
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput, CPU_Unpickler

from ..nn import BaseDecoder, BaseEncoder, BaseDiscriminator
from ..nn.default_architectures import Discriminator_MLP

from typing import Optional

import torch.nn.functional as F


class VAEGAN(VAE):
    """Variational Autoencoder using Adversarial reconstruction loss model.
    
    Args:
        model_config(VAEGANConfig): The Autoencoder configuration seting the main 
            parameters of the model

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        discriminator (BaseDiscriminator): An instance of BaseDiscriminator (inheriting from 
            `torch.nn.Module` which plays the role of discriminator. This argument allows you to use
            your own neural networks architectures if desired. If None is provided, a simple Multi 
            Layer Preception (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default:
            None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: VAEGANConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        discriminator: Optional[BaseDiscriminator]=None
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        if discriminator is None:
            if model_config.latent_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of Adversarial_AE_Config instance "
                    "must be set to 'data_shape' where the shape of the data is "
                    "[mini_batch x data_shape]. Unable to build discriminator automatically."
                )

            self.model_config.discriminator_input_dim = self.model_config.input_dim

            discriminator = Discriminator_MLP(model_config)
            self.model_config.uses_default_discriminator = True

        else:

            self.model_config.uses_default_discriminator = False

        self.set_discriminator(discriminator)

        if self.model_config.reconstruction_layer == -1:
            self.model_config.reconstruction_layer = discriminator.depth

        assert self.model_config.reconstruction_layer <= discriminator.depth, (
                "Ensure that the targeted reconstruction layer ("
                f"{model_config.reconstruction_layer}) is not deeper than the "
                f"discriminator ({discriminator.depth})"
            )

        self.model_name = "VAEGAN"

        assert 0 <= self.model_config.adversarial_loss_scale <= 1, \
            'adversarial_loss_scale must be in [0, 1]'

        assert 0 <= self.model_config.reconstruction_layer <= discriminator.depth, (
            'Cannot use reconstruction layer deeper than discriminator depth '\
            f'({discriminator.depth}). Got ({output_layer_levels})'
        )
        
        self.adversarial_loss_scale = self.model_config.adversarial_loss_scale
        self.reconstruction_layer = self.model_config.reconstruction_layer
        self.margin = self.model_config.margin
        self.equilibrium = self.model_config.equilibrium

    def set_discriminator(self, discriminator: BaseDiscriminator) -> None:
        r"""This method is called to set the metric network outputing the
        :math:`L_{\psi_i}` of the metric matrices

        Args:
            metric (BaseMetric): The metric module that need to be set to the model.

        """
        if not issubclass(type(discriminator), BaseDiscriminator):
            raise BadInheritanceError(
                (
                    "Discriminator must inherit from BaseDiscriminator class from "
                    "pythae.models.base_architectures.BaseDiscriminator. "
                    "Refer to documentation."
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

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z).reconstruction

        z_prior = torch.randn_like(z, device=x.device).requires_grad_(True)

        recon_loss, encoder_loss, decoder_loss, discriminator_loss, update_encoder, update_decoder, update_discriminator = self.loss_function(
            recon_x, x, z, z_prior, mu, log_var
        )

        loss = encoder_loss + decoder_loss + discriminator_loss

        output = ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            encoder_loss=encoder_loss,
            decoder_loss=decoder_loss,
            discriminator_loss=discriminator_loss,
            update_encoder=update_encoder,
            update_decoder=update_decoder,
            update_discriminator=update_discriminator,
            recon_x=recon_x,
            z=z
        )

        return output

    def loss_function(self, recon_x, x, z, z_prior, mu, log_var):

        N = z.shape[0]  # batch size


        # KL between prior and posterior
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)


        # feature maps of true data
        true_discr_layer = self.discriminator(
            x, output_layer_levels=[self.reconstruction_layer])[
                f'embedding_layer_{self.reconstruction_layer}'
            ]

        # feature maps of recon data
        recon_discr_layer = self.discriminator(
            recon_x, output_layer_levels=[self.reconstruction_layer])[
                f'embedding_layer_{self.reconstruction_layer}'
            ]

        # MSE in feature space
        recon_loss = F.mse_loss(
            true_discr_layer.reshape(N, -1),
            recon_discr_layer.reshape(N, -1),
            reduction='none'
        ).sum(dim=-1)

        encoder_loss = KLD + recon_loss

        gen_prior = self.decoder(z_prior).reconstruction

        #x_ = x.clone().detach().requires_grad_(True)
        #recon_x_ = recon_x.clone().detach().requires_grad_(True)
        #gen_prior_ = gen_prior.clone().detach().requires_grad_(True)

        true_adversarial_score = self.discriminator(x).embedding.flatten() 
        gen_adversarial_score = self.discriminator(recon_x).embedding.flatten()
        prior_adversarial_score = self.discriminator(gen_prior).embedding.flatten()

        true_labels = torch.ones(N, requires_grad=False).to(self.device)
        fake_labels = torch.zeros(N, requires_grad=False).to(self.device)

        original_dis_cost = F.binary_cross_entropy(
            true_adversarial_score, true_labels) # original are true
        prior_dis_cost = F.binary_cross_entropy(
            prior_adversarial_score, fake_labels)  # prior is false

        discriminator_loss = (
            (
                original_dis_cost
            )
            +
            (
                prior_dis_cost
            )
            #+
            #(
            #    F.binary_cross_entropy(gen_adversarial_score, fake_labels) # generated are false
            #)
        )

        decoder_loss = (1 - self.adversarial_loss_scale) * recon_loss \
            - self.adversarial_loss_scale * discriminator_loss

        update_encoder = True
        update_discriminator = True
        update_decoder = True

        # margins for training stability
        if original_dis_cost.mean() < self.equilibrium - self.margin or  prior_dis_cost.mean() \
            < self.equilibrium - self.margin:
            update_discriminator = False

        if original_dis_cost.mean() > self.equilibrium + self.margin or prior_dis_cost.mean() \
            > self.equilibrium + self.margin:
            update_decoder = False

        if not update_decoder and not update_discriminator:
            update_discriminator = True
            update_decoder = True

        return (
            (recon_loss).mean(dim=0),
            (encoder_loss).mean(dim=0),
            (decoder_loss).mean(dim=0),
            (discriminator_loss).mean(dim=0),
            update_encoder,
            update_decoder,
            update_discriminator
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

        model_dict = {
            "model_state_dict": deepcopy(self.state_dict()),
        }

        if not self.model_config.uses_default_discriminator:
            with open(os.path.join(model_path, "discriminator.pkl"), "wb") as fp:
                dill.dump(self.discriminator, fp)

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
        model_config = VAEGANConfig.from_json_file(path_to_model_config)

        return model_config

    @classmethod
    def _load_custom_discriminator_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)

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

        model = cls(model_config, encoder=encoder, decoder=decoder, discriminator=discriminator)
        model.load_state_dict(model_weights)

        return model
