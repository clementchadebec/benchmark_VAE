import os

import torch.nn as nn
import torch.nn.functional as F

from ....data.datasets import BaseDataset
from ...base.base_utils import ModelOutput
from ..base import BaseNF
from .pixelcnn_config import PixelCNNConfig
from .utils import MaskedConv2d


class PixelCNN(BaseNF):
    """Pixel CNN model.

    Args:
        model_config (PixelCNNConfig): The PixelCNN model configuration setting the main parameters
            of the model.
    """

    def __init__(self, model_config: PixelCNNConfig):

        BaseNF.__init__(self, model_config=model_config)

        self.model_config = model_config
        self.model_name = "PixelCNN"
        self.net = []

        pad_shape = model_config.kernel_size // 2

        for i in range(model_config.n_layers):
            if i == 0:
                self.net.extend(
                    [
                        nn.Sequential(
                            MaskedConv2d(
                                "A",
                                model_config.input_dim[0],
                                64,
                                model_config.kernel_size,
                                1,
                                pad_shape,
                            ),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                        )
                    ]
                )

            else:
                self.net.extend(
                    [
                        nn.Sequential(
                            MaskedConv2d(
                                "B", 64, 64, model_config.kernel_size, 1, pad_shape
                            ),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                        )
                    ]
                )

        self.net.extend(
            [nn.Conv2d(64, model_config.n_embeddings * model_config.input_dim[0], 1)]
        )

        self.net = nn.Sequential(*self.net)

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is transformed an output image.

        Args:
            inputs (torch.Tensor): An input tensor image. Be carefull it must be in range
                [0-max_channels_values] (i.e. [0-256] for RGB images) and shaped [B x C x H x W].

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        out = self.net(x).reshape(
            x.shape[0],
            self.model_config.n_embeddings,
            self.model_config.input_dim[0],
            x.shape[2],
            x.shape[3],
        )

        loss = F.cross_entropy(out, x.long())

        return ModelOutput(out=out, loss=loss)
