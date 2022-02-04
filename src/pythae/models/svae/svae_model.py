import torch
import torch.distributions as dist
import os
import numpy as np

from ...models import VAE
from .svae_config import SVAEConfig
from .svae_utils import ive
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseEncoder, BaseDecoder
from ..nn.default_architectures import Encoder_SVAE_MLP

from typing import Optional, Union

import torch.nn.functional as F


class SVAE(VAE):
    r"""
    :math:`\mathcal{S}`-VAE model.
    
    Args:
        model_config(SVAEConfig): The Variational Autoencoder configuration seting the main 
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
        model_config: SVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "SVAE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where "
                    "the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_SVAE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)
        
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

        loc, log_concentration = encoder_output.embedding, encoder_output.log_concentration

        # normalize mean
        loc = loc / loc.norm(dim=-1, keepdim=True)

        concentration = torch.nn.functional.softplus(log_concentration)
        z = self._sample_von_mises(loc, concentration)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, loc, concentration, z)

        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, loc, concentration, z):

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

        KLD = self._compute_kl(m=loc.shape[-1], concentration=concentration)

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _compute_kl(self, m, concentration):
        term1 = concentration * (
            ive(m /2, concentration) / (ive(m / 2 - 1, concentration))
        ) # good

        term2 = (
            (m / 2 - 1) * concentration.log() - torch.tensor([2 * np.pi]).to(
                concentration.device
                ).log() * \
            (m / 2) - (ive(m / 2 - 1, concentration)).log() - concentration
        )# good

        term3 = -torch.lgamma(torch.tensor([m / 2]).to(concentration.device)) + \
            torch.tensor([2]).to(concentration.device).log() + \
            torch.tensor([np.pi]).to(concentration.device).log() * (m / 2) # good

        return (term1 + term2 + term3).squeeze(-1)

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _sample_von_mises(self, loc, concentration):

        # Generate uniformly on sphere
        v = torch.randn_like(
            loc[:, 1:]
        )
        v = v / v.norm(dim=-1, keepdim=True)

        w = self._acc_rej_steps(m=loc.shape[-1], k=concentration)

        z = torch.cat((w, (1 - w ** 2).sqrt() * v), dim=-1)

        return self._householder_rotation(loc, z)


    def _householder_rotation(self, loc, z):
        e1 = torch.zeros(z.shape[-1]).to(z.device)
        e1[0] = 1
        u = e1 - loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)

        return z - 2 * u * (u * z).sum(dim=-1, keepdim=True)


    
    def _acc_rej_steps(self, m: int, k: torch.Tensor, device:str='cpu'):

        batch_size = k.shape[0]

        c =  torch.sqrt(4 * k ** 2 + (m - 1) ** 2)

        b = (-2 * k + c) / (m - 1)
        a = (m - 1 + 2 * k + c) / 4
        d = (4 * a * b) /  (1 + b) - (m - 1) * np.log(m - 1)

        d.to(k.device)
        b.to(k.device)

        w = torch.zeros_like(k)

        stopping_mask = torch.ones_like(torch.tensor(b)).type(torch.bool)

        i = 0

        while stopping_mask.sum() > 0:

            i += 1
            
            eps = dist.Beta(
                torch.tensor(0.5 * (m - 1)).type(torch.float),
                torch.tensor(0.5 * (m - 1)).type(torch.float)
            ).sample((batch_size,1)).to(k.device)

            w_ = (1 - (1 + b) * eps) / (1 - (1 - b) * eps)

            t = 2 * a * b / (1 - (1 -b) * eps)

            u =   dist.Uniform(0, 1).sample((batch_size,1)).to(k.device)
            
            acc = (m - 1) * t.log() - t + d > u.log()
            w[acc * stopping_mask] = w_[acc * stopping_mask]

            stopping_mask[acc * stopping_mask] = ~acc[acc * stopping_mask]

        return w

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = SVAEConfig.from_json_file(path_to_model_config)

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
