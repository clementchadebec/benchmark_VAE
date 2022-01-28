import torch
import os

from ...models import AE
from .rae_gp_config import RAE_GP_Config
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput

from ..nn import BaseDecoder, BaseEncoder

from typing import Optional

import torch.nn.functional as F


class RAE_GP(AE):
    """Regularized Autoencoder with gradient penalty model.
    
    Args:
        model_config(RAE_GP_Config): The Autoencoder configuration seting the main parameters of the
            model

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
        model_config: RAE_GP_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        AE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "RAE_GP"


    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded
        
        Args:
            inputs (BaseDataset): An instance of pythae's datasets
            
        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"].requires_grad_(True)

        z = self.encoder(x).embedding
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, gen_reg_loss, embedding_loss = self.loss_function(recon_x, x, z)

        output = ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            gen_reg_loss=gen_reg_loss,
            embedding_loss=embedding_loss,
            recon_x=recon_x,
            z=z
        )

        return output

    def loss_function(self, recon_x, x, z):

        recon_loss = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)

        gen_reg_loss = self._compute_gp(recon_x, x)

        embedding_loss = (0.5 * torch.linalg.norm(z, dim=-1) ** 2)

        return (
            (recon_loss + self.model_config.embedding_weight * embedding_loss \
                + self.model_config.reg_weight * gen_reg_loss).mean(dim=0),
            (recon_loss).mean(dim=0),
            (gen_reg_loss).mean(dim=0),
            (embedding_loss).mean(dim=0)
        )

    def _compute_gp(self, recon_x, x):
        grads = torch.autograd.grad(
            outputs=recon_x,
            inputs=x,
            grad_outputs=torch.ones_like(recon_x).to(self.device),
            create_graph=True,
            retain_graph=True
        )[0].reshape(recon_x.shape[0], -1)

        return (grads.norm(dim=-1) ** 2)


    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = RAE_GP_Config.from_json_file(path_to_model_config)

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
