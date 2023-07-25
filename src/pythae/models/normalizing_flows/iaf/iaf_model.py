import os

import numpy as np
import torch
import torch.nn as nn

from pythae.models.base.base_utils import ModelOutput

from ..base import BaseNF
from ..layers import BatchNorm
from ..made import MADE, MADEConfig
from .iaf_config import IAFConfig


class IAF(BaseNF):
    """Inverse Autoregressive Flow.

    Args:
        model_config (IAFConfig): The IAF model configuration setting the main parameters of the
            model.
    """

    def __init__(self, model_config: IAFConfig):

        BaseNF.__init__(self, model_config=model_config)

        self.net = []
        self.m = {}
        self.model_config = model_config
        self.input_dim = np.prod(model_config.input_dim)
        self.hidden_size = model_config.hidden_size
        self.model_name = "IAF"

        made_config = MADEConfig(
            input_dim=(self.input_dim,),
            output_dim=(self.input_dim,),
            hidden_sizes=[self.hidden_size] * self.model_config.n_hidden_in_made,
            degrees_ordering="sequential",
        )

        for i in range(model_config.n_made_blocks):
            self.net.extend([MADE(made_config)])
            if self.model_config.include_batch_norm:
                self.net.extend([BatchNorm(self.input_dim)])

        self.net = nn.ModuleList(self.net)

    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """The input data is transformed toward the prior (f^{-1})

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = x.reshape(x.shape[0], -1)
        sum_log_abs_det_jac = torch.zeros(x.shape[0]).to(x.device)

        for layer in self.net:
            if layer.__class__.__name__ == "MADE":
                y = torch.zeros_like(x)
                for i in range(self.input_dim):
                    layer_out = layer(y.clone())

                    m, s = layer_out.mu, layer_out.log_var

                    y[:, i] = (x[:, i] - m[:, i]) * (-s[:, i]).exp()

                    sum_log_abs_det_jac += -s[:, i]

                x = y
            else:
                layer_out = layer(x)
                x = layer_out.out
                sum_log_abs_det_jac += layer_out.log_abs_det_jac

            x = x.flip(dims=(1,))

        return ModelOutput(out=x, log_abs_det_jac=sum_log_abs_det_jac)

    def inverse(self, y: torch.Tensor, **kwargs) -> ModelOutput:
        """The prior is transformed toward the input data (f)

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        y = y.reshape(y.shape[0], -1)
        sum_log_abs_det_jac = torch.zeros(y.shape[0]).to(y.device)

        for layer in self.net[::-1]:
            y = y.flip(dims=(1,))
            if layer.__class__.__name__ == "MADE":
                layer_out = layer(y)
                m, s = layer_out.mu, layer_out.log_var
                y = y * s.exp() + m
                sum_log_abs_det_jac += s.sum(dim=-1)

            else:
                layer_out = layer.inverse(y)
                y = layer_out.out
                sum_log_abs_det_jac += layer_out.log_abs_det_jac

        return ModelOutput(out=y, log_abs_det_jac=sum_log_abs_det_jac)
