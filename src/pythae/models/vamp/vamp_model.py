import torch
import torch.nn as nn
import numpy as np
import os

from ...models import VAE
from .vamp_config import VAMPConfig
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseEncoder, BaseDecoder
from ..nn.default_architectures import Encoder_VAE_MLP

from typing import Optional

import torch.nn.functional as F


class VAMP(VAE):
    """Variational Mixture of Posteriors (VAMP) VAE model
    
    Args:
        model_config(VAEConfig): The Variational Autoencoder configuration seting the main 
        parameters of the model

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
        model_config: VAMPConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "VAMP"

        self.number_components = model_config.number_components

        if model_config.input_dim is None:
            raise AttributeError("Provide input dim to build pseudo input network")

        linear_layer = nn.Linear(
                model_config.number_components, int(np.prod(model_config.input_dim))
            )

        self.pseudo_inputs = nn.Sequential(
            linear_layer,
            nn.Hardtanh(0.0, 1.0),
        )

        # init weights
        #linear_layer.weight.data.normal_(0, 0.0)

        self.idle_input = torch.eye(
            model_config.number_components, requires_grad=False
        ).to(self.device)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training datasat with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        # need to put model in train mode to make it work. If you have a solution to this issue 
        # please open a pull request at (https://github.com/clementchadebec/benchmark_VAE/pulls)
        self.train()
        x = inputs["data"]

        encoder_output = self.encoder(x)

        # we bound log_var to avoid unbounded optim
        mu, log_var = encoder_output.embedding, torch.tanh(encoder_output.log_covariance)

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)

        recon_x = self.decoder(z)["reconstruction"]

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

        log_p_z = self._log_p_z(z)

        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / torch.exp(log_var))).sum(
            dim=1
        )
        KLD = -(log_p_z - log_q_z)

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _log_p_z(self, z):
        """Computation of the log prob of the VAMP"""

        C = self.number_components

        x = self.pseudo_inputs(self.idle_input.to(self.device)).reshape(
            (C,) + self.model_config.input_dim
        )

        # we bound log_var to avoid unbounded optim
        encoder_output = self.encoder(x)
        prior_mu, prior_log_var = (
            encoder_output.embedding,
            torch.tanh(encoder_output.log_covariance),
        )

        z_expand = z.unsqueeze(1)
        prior_mu = prior_mu.unsqueeze(0)
        prior_log_var = prior_log_var.unsqueeze(0)

        log_p_z = torch.sum(
            -0.5 * (prior_log_var + (z_expand - prior_mu) ** 2) / prior_log_var.exp(),
            dim=2,
        ) - torch.log(torch.tensor(C).type(torch.float))

        log_p_z = torch.logsumexp(log_p_z, dim=1)

        return log_p_z

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
        model_config = VAMPConfig.from_json_file(path_to_model_config)

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
