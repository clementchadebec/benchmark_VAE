import os
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from ....data.datasets import BaseDataset
from ...auto_model import AutoConfig
from ...base.base_config import EnvironmentConfig
from ...base.base_utils import ModelOutput
from .base_nf_config import BaseNFConfig


class BaseNF(nn.Module):
    """Base Class from Normalizing flows

    Args:
        model_config (BaseNFConfig): The configuration setting the main parameters of the
            model.
    """

    def __init__(self, model_config: BaseNFConfig):

        nn.Module.__init__(self)

        if model_config.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of MADEConfig instance must be set to 'data_shape' "
                "where the shape of the data is (C, H, W ..)]. Unable to build network"
                "automatically"
            )

        self.model_config = model_config
        self.input_dim = np.prod(model_config.input_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """Main forward pass mapping the data towards the prior
        This function should output a :class:`~pythae.models.base.base_utils.ModelOutput` instance
        gathering all the model outputs

        Args:
            x (torch.Tensor): The training data.

        Returns:
            ModelOutput: A ModelOutput instance providing the outputs of the model.
        """
        raise NotImplementedError()

    def inverse(self, y: torch.Tensor, **kwargs) -> ModelOutput:
        """Main inverse pass mapping the prior toward the data
        This function should output a :class:`~pythae.models.base.base_utils.ModelOutput` instance
        gathering all the model outputs

        Args:
            inputs (torch.Tensor): Data from the prior.

        Returns:
            ModelOutput: A ModelOutput instance providing the outputs of the model.
        """
        raise NotImplementedError()

    def update(self):
        """Method that allows model update during the training (at the end of a training epoch)

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """

    def save(self, dir_path):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file.

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """

        env_spec = EnvironmentConfig(
            python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"
        )
        model_dict = {"model_state_dict": deepcopy(self.state_dict())}

        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)

            except (FileNotFoundError, TypeError) as e:
                raise e

        env_spec.save_json(dir_path, "environment")
        self.model_config.save_json(dir_path, "model_config")

        torch.save(model_dict, os.path.join(dir_path, "model.pt"))

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = AutoConfig.from_json_file(path_to_model_config)

        return model_config

    @classmethod
    def _load_model_weights_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model.pt" not in file_list:
            raise FileNotFoundError(
                f"Missing model weights file ('model.pt') file in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_weights = os.path.join(dir_path, "model.pt")

        try:
            model_weights = torch.load(path_to_model_weights, map_location="cpu")

        except RuntimeError:
            RuntimeError(
                "Enable to load model weights. Ensure they are saves in a '.pt' format."
            )

        if "model_state_dict" not in model_weights.keys():
            raise KeyError(
                "Model state dict is not available in 'model.pt' file. Got keys:"
                f"{model_weights.keys()}"
            )

        model_weights = model_weights["model_state_dict"]

        return model_weights

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided
        """

        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)
        model = cls(model_config)
        model.load_state_dict(model_weights)

        return model


class NFModel(nn.Module):
    """Class wrapping the normalizing flows so it can articulate with
    :class:`~pythae.trainers.BaseTrainer`
    """

    def __init__(self, prior: torch.distributions, flow: BaseNF):
        nn.Module.__init__(self)
        self.prior = prior
        self.flow = flow

    @property
    def model_config(self):
        return self.flow.model_config

    @property
    def model_name(self):
        return self.flow.model_name

    def forward(self, x: BaseDataset, **kwargs):
        x = x["data"]

        flow_output = self.flow(x, **kwargs)

        y = flow_output.out
        log_abs_det_jac = flow_output.log_abs_det_jac

        log_prob_prior = self.prior.log_prob(y).reshape(y.shape[0])

        output = ModelOutput(loss=-(log_prob_prior + log_abs_det_jac).sum())

        return output

    def update(self):
        self.flow.update()

    def save(self, dir_path):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file.
        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """

        self.flow.save(dir_path=dir_path)
