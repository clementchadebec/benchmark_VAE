import torch
import torch.nn as nn
import numpy as np
import os

from copy import deepcopy
from pythae.models.base.base_utils import ModelOutput

from ....data.datasets import BaseDataset
from ..layers import BatchNorm
from ..base import BaseNF
from ..made import MADE, MADEConfig
from .maf_config import MAFConfig

class MAF(BaseNF):
    """Masked Autoregressive Flow
    
    Args:
        model_config (MAFConfig): The MAF model configuration setting the main parameters of the 
            model
    """

    def __init__(
        self,
        model_config: MADEConfig):

        BaseNF.__init__(self, model_config=model_config)
        
        self.net = []
        self.m = {}
        self.model_config = model_config
        self.input_dim = np.prod(model_config.input_dim)
        self.hidden_size = model_config.hidden_size

        if model_config.input_dim is None:
            raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of MADEConfig instance must be set to 'data_shape' "
                    "where the shape of the data is (C, H, W ..)]. Unable to build network"
                    "automatically"
                )

        self.net = []

        made_config = MADEConfig(
                input_dim=(self.input_dim, ),
                output_dim=(self.input_dim, ),
                hidden_sizes=[self.hidden_size]*self.model_config.n_hidden_in_made,
                degrees_ordering="sequential"
            )

        for i in range(model_config.n_made_blocks):
            self.net.extend([
                MADE(made_config),
                BatchNorm(self.input_dim)
                ]
            )

        self.net = nn.ModuleList(self.net)

    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """The input data is transformed toward the prior

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        sum_log_abs_det_jac = 0

        for layer in self.net:
            layer_out = layer(x)
            x = layer_out.out.flip(dim=-1)
            sum_log_abs_det_jac += layer_out.log_abs_det_jac

        return ModelOutput(
            out=x,
            log_abs_det_jac=sum_log_abs_det_jac
        )

    def inverse(self, y: torch.Tensor, **kwargs) -> ModelOutput:
        """The prior is transformed toward the input data

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        sum_log_abs_det_jac = 0

        for layer in self.net[::-1]:
            layer_out = layer.inverse(y)
            y = layer_out.out
            sum_log_abs_det_jac += layer_out.log_abs_det_jac

        return ModelOutput(
            out=y,
            log_abs_det_jac=sum_log_abs_det_jac
        )

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = MAFConfig.from_json_file(path_to_model_config)

        return model_config
