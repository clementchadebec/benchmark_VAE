from ..base import BaseAE
from .ae_config import VAEConfig
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOuput

import torch.nn.functional as F

class VAE(BaseAE):
    """Vanilla Autoencoder model.
    
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
        model_config: AEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None
    ):

        BaseAE.__init__(
            self, model_config=model_config, encoder=encoder, decoder=decoder
            )


    def forward(self, x):
        """
        The VAE model
        """

        encoder_output = self.encode(x)

        mu, log_var = encoder_output['']

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)