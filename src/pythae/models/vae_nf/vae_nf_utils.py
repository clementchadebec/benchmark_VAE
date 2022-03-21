"""Code adapted from https://github.com/mrsalehi/stupid-simple-norm-flow.
2 simple flow classes that can be combined"""

import torch
import torch.nn as nn
from ..base.base_utils import ModelOutput
import torch.nn.functional as F
import math

from .vae_nf_config import VAE_NF_Config

class PlanarFlow(nn.Module):
    def __init__(self, dim: int, activation:str='tanh'):
        f"""
        Planar flow instance.

        Args:
            dim (int): The dimension the flows lives in.

            activation (str): The activation function to be applied in the flow. 
                Possible choices are {[key for key in ACTIVATION.keys()]}. 
                Default: 'tanh'.
        """

        assert activation in ACTIVATION.keys(), (
            f"`{activation}` function is not handled. Possible activation: "
            f"{[key for key in ACTIVATION.keys()]}"
        )

        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.empty(dim))
        self.b = nn.Parameter(torch.empty(1))
        self.u = nn.Parameter(torch.empty(dim))
        self.activation = ACTIVATION[activation]
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]

        nn.init.normal_(self.w)
        nn.init.normal_(self.u)
        nn.init.normal_(self.b)

    def forward(self, z: torch.Tensor):
        """
        Forward method of the flow.

        Args:
            z (torch.Tensor): The input tensor.

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters.
        """
        lin = (z @ self.w + self.b).unsqueeze(1)  # [Bx1]
        f = z + self.u * self.activation(lin)  # [Bxdim]
        phi = self.activation_derivative(lin) * self.w  # [Bxdim]
        log_det = torch.log(torch.abs(1 + phi @ self.u) + 1e-4) # [B]
        
        output = ModelOutput(
            z=f,
            log_det=log_det
        )

        return output


class RadialFlow(nn.Module):
    def __init__(self, dim: int, activation: str = 'tanh'):
        f"""
        Radial flow instance.

        Args:
            dim (int): The dimension the flows lives in.

            activation (str): The activation function to be applied in the flow. 
                Possible choices are {[key for key in ACTIVATION.keys()]}. 
                Default: 'tanh'.
        """
        assert activation in ACTIVATION.keys(), (
            f"`{activation}` function is not handled. Possible activation: "
            f"{[key for key in ACTIVATION.keys()]}"
        )

        super().__init__()

        self.z0 = nn.Parameter(torch.empty(dim))
        self.log_alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))
        self.activation = ACTIVATION[activation]
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]
        self.dim = dim

        nn.init.normal_(self.z0) 
        nn.init.normal_(self.log_alpha)
        nn.init.normal_(self.beta)


    def forward(self, z: torch.Tensor):
        """
        Forward method of the flow.

        Args:
            z (torch.Tensor): The input tensor.

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters.
        """
        z_sub = z - self.z0
        alpha = torch.exp(self.log_alpha)
        r = torch.norm(z_sub, dim=-1, keepdim=True) # [Bx1]
        h = 1 / (alpha + r) # [Bx1]
        f = z + self.beta * h * z_sub #[Bxdim]
        log_det = (self.dim - 1) * torch.log(1 + self.beta * h) + \
            torch.log(1 + self.beta * h + self.beta - self.beta * r / (alpha + r) ** 2)

        output = ModelOutput(
            z=f,
            log_det=log_det
        )

        return output

ACTIVATION = {
    "elu": F.elu,
    "tanh": torch.tanh,
    "linear": lambda x: x
}

ACTIVATION_DERIVATIVES = {
    "elu": lambda x: torch.ones_like(x) * (x >= 0) + torch.exp(x) * (x < 0),
    "tanh": lambda x: 1 - torch.tanh(x) ** 2,
    "linear": lambda x: 1
}