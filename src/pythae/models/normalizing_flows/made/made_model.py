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

        hidden_sizes = [self.input_dim] + model_config.hidden_sizes + [self.output_dim]

        masks = self._make_mask(ordering=self.model_config.degrees_ordering)

        for inp, out, mask in zip(hidden_sizes[:-1], hidden_sizes[1:-1], masks[:-1]):

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

        if ordering == "sequential":
            self.m[-1] = torch.arange(self.input_dim)
            for i in range(len(self.hidden_sizes)):
                self.m[i] = torch.arange(self.hidden_sizes[i]) % (self.input_dim - 1)

        else:
            self.m[-1] = torch.randperm(self.input_dim)
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

    def forward(self, x: torch.tensor, **kwargs) -> ModelOutput:
        """The input data is transformed toward the prior

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """
        net_output = self.net(x.reshape(x.shape[0], -1))

        mu = net_output[:, : self.input_dim]
        log_var = net_output[:, self.input_dim :]

        return ModelOutput(mu=mu, log_var=log_var)

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = MADEConfig.from_json_file(path_to_model_config)

        return model_config
