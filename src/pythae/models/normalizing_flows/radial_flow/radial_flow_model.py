import os

import torch
import torch.nn as nn

from ...base.base_utils import ModelOutput
from ..base import BaseNF
from .radial_flow_config import RadialFlowConfig


class RadialFlow(BaseNF):
    """Radial Flow model.

    Args:
        model_config (RadialFlowConfig): The RadialFlow model configuration setting the main parameters of
            the model.
    """

    def __init__(self, model_config: RadialFlowConfig):

        BaseNF.__init__(self, model_config)

        self.x0 = nn.Parameter(torch.randn(1, self.input_dim))
        self.log_alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))
        self.model_name = "RadialFlow"

        nn.init.normal_(self.x0)
        nn.init.normal_(self.log_alpha)
        nn.init.normal_(self.beta)

    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """The input data is transformed toward the prior

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = x.reshape(x.shape[0], -1)
        x_sub = x - self.x0
        alpha = torch.exp(self.log_alpha)
        beta = -alpha + torch.log(1 + self.beta.exp())  # ensure invertibility
        r = torch.norm(x_sub, dim=-1, keepdim=True)  # [Bx1]
        h = 1 / (alpha + r)  # [Bx1]
        f = x + beta * h * x_sub  # [Bxdim]
        log_det = (self.input_dim - 1) * torch.log(1 + beta * h) + torch.log(
            1 + beta * h - beta * r / (alpha + r) ** 2
        )

        output = ModelOutput(out=f, log_abs_det_jac=log_det.squeeze())

        return output
