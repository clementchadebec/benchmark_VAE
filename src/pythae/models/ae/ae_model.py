from ..base import BaseAE
from .ae_config import AEConfig
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOuput

from pyraug.models.nn import BaseDecoder, BaseEncoder

from typing import Optional

import torch.nn.functional as F

class AE(BaseAE):
    """Vanilla Autoencoder model.
    
    Args:
        model_config(AEConfig): The Autoencoder configuration seting the main parameters of the
            model

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
        model_config: AEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None
    ):

        BaseAE.__init__(
            self, model_config=model_config, encoder=encoder, decoder=decoder
            )
    
    def forward(self, inputs: BaseDataset) -> ModelOuput:
        """The input data is encoded and decoded
        
        Args:
            inputs (~pyraug.data.datassets.BaseDataset): An instance of pyraug's datasets
            
        Returns:
            output (ModelOutput): An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]

        z = self.encoder(x)
        recon_x = self.decoder(z)

        loss = self.loss_function(recon_x, x)

        output = ModelOuput(
            loss=loss,
            recon_x=recon_x,
            z=z
        )

        return output


    def loss_function(self, recon_x, x):

        MSE = F.mse_loss(recon_x.reshape(shape[0], -1), x.reshape(x.shape[0], -1), reduction='sum')
        return MSE