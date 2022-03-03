import torch
import os
import numpy as np

from ...models import VAE
from .beta_tc_vae_config import BetaTCVAEConfig
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseEncoder, BaseDecoder
from ..nn.default_architectures import Encoder_VAE_MLP

from typing import Optional

import torch.nn.functional as F


class BetaTCVAE(VAE):
    r"""
    :math:`\beta`-TCVAE model.
    
    Args:
        model_config(BetaTCVAEConfig): The Variational Autoencoder configuration seting the main 
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
        model_config: BetaTCVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "BetaTCVAE"
        self.alpha = model_config.alpha 
        self.beta = model_config.beta
        self.gamma = model_config.gamma
        self.use_mss = model_config.use_mss


    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training datasat with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        dataset_size = epoch = kwargs.pop('dataset_size', x.shape[0])

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z, dataset_size)

        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z, dataset_size):

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

        log_q_z_given_x = self._compute_log_gauss_density(z, mu, log_var).sum(dim=-1) #[B]

        log_prior = self._compute_log_gauss_density(
            z,
            torch.zeros_like(z),
            torch.zeros_like(z)
        ).sum(dim=-1) # [B]

        log_q_batch_perm = self._compute_log_gauss_density(
            z.reshape(z.shape[0], 1, -1),
            mu.reshape(1, z.shape[0], -1),
            log_var.reshape(1, z.shape[0], -1)
        ) # [B x B x Latent_dim]

        if self.use_mss:
            logiw_mat = self._log_importance_weight_matrix(z.shape[0], dataset_size).to(z.device)
            log_q_z = torch.logsumexp(logiw_mat + log_q_batch_perm.sum(dim=-1), dim=-1) # MMS [B]
            log_prod_q_z =  (
                torch.logsumexp(
                    logiw_mat.reshape(z.shape[0], z.shape[0], -1) + log_q_batch_perm, dim=1
                )
            ).sum(dim=-1) # MMS [B]

        else:
            log_q_z = torch.logsumexp(log_q_batch_perm.sum(dim=-1), dim=-1) - torch.log(
                torch.tensor([z.shape[0]*dataset_size]).to(z.device)) # MWS [B]
            log_prod_q_z =  (
                torch.logsumexp(log_q_batch_perm, dim=1) - torch.log(
                torch.tensor([z.shape[0]*dataset_size]).to(z.device))
                ).sum(dim=-1) # MWS [B]


        mutual_info_loss = log_q_z_given_x - log_q_z
        TC_loss = log_q_z - log_prod_q_z
        dimension_wise_KL = log_prod_q_z - log_prior


        return (
            (
                recon_loss
                + self.alpha * mutual_info_loss
                + self.beta * TC_loss
                + self.gamma * dimension_wise_KL
            ).mean(dim=0),
            recon_loss.mean(dim=0),
            (
                self.alpha * mutual_info_loss
                + self.beta * TC_loss
                + self.gamma * dimension_wise_KL
            ).mean(dim=0)
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _compute_log_gauss_density(self, z, mu, log_var):
        """element-wise computation"""
        return (
            -0.5 * (
                torch.log(torch.tensor([np.pi]).to(z.device))
                + log_var
                + (z - mu)**2 * torch.exp(-log_var)
            )
        )

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        """Compute importance weigth matrix for MSS
        Code from (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
        """

        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = BetaTCVAEConfig.from_json_file(path_to_model_config)

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
