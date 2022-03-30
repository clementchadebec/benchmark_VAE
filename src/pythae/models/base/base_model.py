import os
from copy import deepcopy
from typing import Optional

import dill
import torch
import torch.nn as nn

from ...customexception import BadInheritanceError
from ...data.datasets import BaseDataset
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Decoder_AE_MLP, Encoder_AE_MLP
from .base_config import BaseAEConfig
from .base_utils import CPU_Unpickler, ModelOutput


class BaseAE(nn.Module):
    """Base class for Autoencoder based models.

    Args:
        model_config (BaseAEConfig): An instance of BaseAEConfig in which any model's parameters is
            made available.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: BaseAEConfig,
        encoder: Optional[BaseDecoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        nn.Module.__init__(self)

        self.model_name = "BaseAE"

        self.input_dim = model_config.input_dim
        self.latent_dim = model_config.latent_dim

        self.model_config = model_config

        if decoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where "
                    "the shape of the data is (C, H, W ..)]. Unable to build decoder"
                    "automatically"
                )

            decoder = Decoder_AE_MLP(model_config)
            self.model_config.uses_default_decoder = True

        else:
            self.model_config.uses_default_decoder = False

        self.set_decoder(decoder)

        self.device = None

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """Main forward pass outputing the VAE outputs
        This function should output a :class:`~pythae.models.base.base_utils.ModelOutput` instance
        gathering all the model outputs

        Args:
            inputs (BaseDataset): The training data with labels, masks etc...

        Returns:
            ModelOutput: A ModelOutput instance providing the outputs of the model.

        .. note::
            The loss must be computed in this forward pass and accessed through
            ``loss = model_output.loss``"""
        raise NotImplementedError()

    def update(self):
        """Method that allows model update during the training (at the end of a training epoch)

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """
        pass

    def save(self, dir_path):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file. If the
        model to save used custom encoder (resp. decoder) provided by the user, these are also 
        saved as ``decoder.pkl`` (resp. ``decoder.pkl``).

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """

        model_path = dir_path

        model_dict = {"model_state_dict": deepcopy(self.state_dict())}

        if not os.path.exists(model_path):
            try:
                os.makedirs(model_path)

            except FileNotFoundError as e:
                raise e

        self.model_config.save_json(model_path, "model_config")

        # only save .pkl if custom architecture provided
        if not self.model_config.uses_default_encoder:
            with open(os.path.join(model_path, "encoder.pkl"), "wb") as fp:
                dill.dump(self.encoder, fp)

        if not self.model_config.uses_default_decoder:
            with open(os.path.join(model_path, "decoder.pkl"), "wb") as fp:
                dill.dump(self.decoder, fp)

        torch.save(model_dict, os.path.join(model_path, "model.pt"))

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = BaseAEConfig.from_json_file(path_to_model_config)

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
    def _load_custom_encoder_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)
        if "encoder.pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing encoder pkl file ('encoder.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom encoders."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "encoder.pkl"), "rb") as fp:
                encoder = CPU_Unpickler(fp).load()

        return encoder

    @classmethod
    def _load_custom_decoder_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)

        if "decoder.pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing decoder pkl file ('decoder.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom decoders."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "decoder.pkl"), "rb") as fp:
                decoder = CPU_Unpickler(fp).load()

        return decoder

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)

        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)

        else:
            encoder = None

        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)

        else:
            decoder = None

        model = cls(model_config, encoder=encoder, decoder=decoder)
        model.load_state_dict(model_weights)

        return model

    def set_encoder(self, encoder: BaseEncoder) -> None:
        """Set the encoder of the model"""
        if not issubclass(type(encoder), BaseEncoder):
            raise BadInheritanceError(
                (
                    "Encoder must inherit from BaseEncoder class from "
                    "pythae.models.base_architectures.BaseEncoder. Refer to documentation."
                )
            )
        self.encoder = encoder

    def set_decoder(self, decoder: BaseDecoder) -> None:
        """Set the decoder of the model"""
        if not issubclass(type(decoder), BaseDecoder):
            raise BadInheritanceError(
                (
                    "Decoder must inherit from BaseDecoder class from "
                    "pythae.models.base_architectures.BaseDecoder. Refer to documentation."
                )
            )
        self.decoder = decoder
