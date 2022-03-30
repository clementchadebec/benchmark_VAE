import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from pythae.models.base.base_utils import ModelOutput

from ....data.datasets import BaseDataset
from ..base import BaseNF
from ..layers import BatchNorm
from ..made import MADE, MADEConfig
from ..maf.maf_model import MAF
from .iaf_config import IAFConfig


class IAF(MAF):
    """Inverse Autoregressive Flow.
    
    Args:
        model_config (IAFConfig): The IAF model configuration setting the main parameters of the 
            model
    """

    def __init__(self, model_config: IAFConfig):

        MAF.__init__(self, model_config=model_config)

        self.model_name = "IAF"

    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """The input data is transformed toward the prior

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        sum_log_abs_det_jac = torch.zeros(x.shape[0]).to(x.device)

        for layer in self.net:
            if layer.__class__.__name__ == "MADE":
                layer_out = layer.inverse(x)
            else:
                layer_out = layer(x)
            x = layer_out.out.flip(dims=(1,))
            sum_log_abs_det_jac += layer_out.log_abs_det_jac

        return ModelOutput(out=x, log_abs_det_jac=sum_log_abs_det_jac)

    def inverse(self, y: torch.Tensor, **kwargs) -> ModelOutput:
        """The prior is transformed toward the input data

        Args:
            inputs (torch.Tensor): An input tensor

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        sum_log_abs_det_jac = torch.zeros(y.shape[0]).to(y.device)

        for layer in self.net[::-1]:
            y = y.flip(dims=(1,))
            if layer.__class__.__name__ == "MADE":
                layer_out = layer(y)
            else:
                layer_out = layer.inverse(y)
            y = layer_out.out
            sum_log_abs_det_jac += layer_out.log_abs_det_jac

        return ModelOutput(out=y, log_abs_det_jac=sum_log_abs_det_jac)

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = IAFConfig.from_json_file(path_to_model_config)

        return model_config
