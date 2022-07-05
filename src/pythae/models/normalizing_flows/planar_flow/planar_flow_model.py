import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base.base_utils import ModelOutput
from ..base import BaseNF
from .planar_flow_config import PlanarFlowConfig

ACTIVATION = {"elu": F.elu, "tanh": torch.tanh, "linear": lambda x: x}

ACTIVATION_DERIVATIVES = {
    "elu": lambda x: torch.ones_like(x) * (x >= 0) + torch.exp(x) * (x < 0),
    "tanh": lambda x: 1 - torch.tanh(x) ** 2,
    "linear": lambda x: 1,
}


class PlanarFlow(BaseNF):
    """Planar Flow model.

    Args:
        model_config (PlanarFlowConfig): The PlanarFlow model configuration setting the main parameters of
            the model.
    """

    def __init__(self, model_config: PlanarFlowConfig):

        BaseNF.__init__(self, model_config)

        self.w = nn.Parameter(torch.randn(1, self.input_dim))
        self.u = nn.Parameter(torch.randn(1, self.input_dim))
        self.b = nn.Parameter(torch.randn(1))
        self.activation = ACTIVATION[model_config.activation]
        self.activation_derivative = ACTIVATION_DERIVATIVES[model_config.activation]
        self.model_name = "PlanarFlow"

        nn.init.normal_(self.w)
        nn.init.normal_(self.u)
        nn.init.normal_(self.b)

    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """The input data is transformed toward the prior

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        x = x.reshape(x.shape[0], -1)
        lin = x @ self.w.T + self.b  # [B x 1]
        f = x + self.u * self.activation(lin)  # [B x input_dim]
        phi = self.activation_derivative(lin) @ self.w  # [B x input_dim]
        log_det = torch.log(torch.abs(1 + phi @ self.u.T) + 1e-4).squeeze()  # [B]

        output = ModelOutput(out=f, log_abs_det_jac=log_det)

        return output
