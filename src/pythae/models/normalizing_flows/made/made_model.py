import os

import numpy as np
import torch
import torch.nn as nn

from pythae.models.base.base_utils import ModelOutput

from ..base import BaseNF
from ..layers import MaskedLinear
from .made_config import MADEConfig


class MADE(BaseNF):
    """Masked Autoencoder model

    Args:
        model_config (MADEConfig): The MADE model configuration setting the main parameters of the
            model
    """

    def __init__(self, model_config: MADEConfig):

        BaseNF.__init__(self, model_config=model_config)

        self.net = []
        self.m = {}
        self.model_config = model_config
        self.input_dim = np.prod(model_config.input_dim)
        self.output_dim = np.prod(model_config.output_dim)
        self.hidden_sizes = model_config.hidden_sizes
        self.context_dim = np.prod(model_config.context_dim)
        self.model_name = "MADE"

        if model_config.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of MADEConfig instance must be set to 'data_shape' "
                "where the shape of the data is (C, H, W ..)]. Unable to build network"
                "automatically"
            )

        if model_config.output_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'output_dim' parameter of MADEConfig instance must be set to 'data_shape' "
                "where the shape of the data is (C, H, W ..)]. Unable to build network"
                "automatically"
            )

        hidden_sizes = model_config.hidden_sizes + [self.output_dim]

        masks = self._make_mask(ordering=self.model_config.degrees_ordering)

        self.context_input_layer = MaskedLinear(
            self.input_dim,
            hidden_sizes[0],
            masks[0],
            context_features=self.context_dim,
        )

        for inp, out, mask in zip(hidden_sizes[:-1], hidden_sizes[1:-1], masks[1:-1]):

            self.net.extend([MaskedLinear(inp, out, mask), nn.ReLU()])

        # outputs mean and logvar
        self.net.extend(
            [
                MaskedLinear(
                    self.hidden_sizes[-1], 2 * self.output_dim, masks[-1].repeat(2, 1)
                )
            ]
        )

        self.net = nn.Sequential(*self.net)

    def _make_mask(self, ordering="sequential"):

        # Get degrees for mask creation
        self.m[-1] = torch.arange(1, self.input_dim + 1)
        if ordering == "sequential":
            for i in range(len(self.hidden_sizes)):
                min_deg = min(torch.min(self.m[i - 1]), self.input_dim - 1)
                self.m[i] = np.maximum(
                    min_deg,
                    np.ceil(
                        np.arange(1, self.hidden_sizes[i] + 1)
                        * (self.input_dim - 1)
                        / float(self.hidden_sizes[i] + 1)
                    ).astype(np.int32),
                )

        else:
            idx = torch.randperm(self.input_dim)
            self.m[-1] = self.m[-1][idx]
            for i in range(len(self.hidden_sizes)):
                self.m[i] = torch.randint(
                    self.m[-1].min(), self.input_dim - 1, (self.hidden_sizes[i],)
                )

        masks = []
        for i in range(len(self.hidden_sizes)):
            masks += [(self.m[i].unsqueeze(-1) >= self.m[i - 1].unsqueeze(0)).float()]

        masks.append(
            (
                self.m[len(self.hidden_sizes) - 1].unsqueeze(0)
                < self.m[-1].unsqueeze(-1)
            ).float()
        )

        return masks

    def forward(self, x: torch.tensor, h=None, **kwargs) -> ModelOutput:
        """The input data is transformed toward the prior

        Args:
            x (torch.Tensor): An input tensor.
            h (torch.Tensor): The context tensor. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        out = self.context_input_layer(x.reshape(x.shape[0], -1), h)
        net_output = self.net(out)

        mu = net_output[:, : self.input_dim]
        log_var = net_output[:, self.input_dim :]

        return ModelOutput(mu=mu, log_var=log_var)
