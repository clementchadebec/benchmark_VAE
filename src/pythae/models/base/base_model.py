import inspect
import logging
import os
import shutil
import sys
import tempfile
import warnings
from copy import deepcopy
from http.cookiejar import LoadError
from typing import Optional

import cloudpickle
import torch
import torch.nn as nn

from ...customexception import BadInheritanceError
from ...data.datasets import BaseDataset, DatasetOutput
from ..auto_model import AutoConfig
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Decoder_AE_MLP
from .base_config import BaseAEConfig, EnvironmentConfig
from .base_utils import (
    CPU_Unpickler,
    ModelOutput,
    hf_hub_is_available,
    model_card_template,
)

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


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

    def reconstruct(self, inputs: torch.Tensor):
        """This function returns the reconstructions of given input data.

        Args:
            inputs (torch.Tensor): The inputs data to be reconstructed of shape [B x input_dim]
            ending_inputs (torch.Tensor): The starting inputs in the interpolation of shape

        Returns:
            torch.Tensor: A tensor of shape [B x input_dim] containing the reconstructed samples.
        """
        return self(DatasetOutput(data=inputs)).recon_x

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the embeddings of the input data.

        Args:
            inputs (torch.Tensor): The input data to be embedded, of shape [B x input_dim].

        Returns:
            torch.Tensor: A tensor of shape [B x latent_dim] containing the embeddings.
        """
        return self(DatasetOutput(data=inputs)).z

    def predict(self, inputs: torch.Tensor) -> ModelOutput:
        """The input data is encoded and decoded without computing loss

        Args:
            inputs (torch.Tensor): The input data to be reconstructed, as well as to generate the embedding.

        Returns:
            ModelOutput: An instance of ModelOutput containing reconstruction and embedding
        """
        z = self.encoder(inputs).embedding
        recon_x = self.decoder(z)["reconstruction"]

        output = ModelOutput(
            recon_x=recon_x,
            embedding=z,
        )

        return output

    def interpolate(
        self,
        starting_inputs: torch.Tensor,
        ending_inputs: torch.Tensor,
        granularity: int = 10,
    ):
        """This function performs a linear interpolation in the latent space of the autoencoder
        from starting inputs to ending inputs. It returns the interpolation trajectories.

        Args:
            starting_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            ending_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            granularity (int): The granularity of the interpolation.

        Returns:
            torch.Tensor: A tensor of shape [B x granularity x input_dim] containing the
            interpolation trajectories.
        """
        assert starting_inputs.shape[0] == ending_inputs.shape[0], (
            "The number of starting_inputs should equal the number of ending_inputs. Got "
            f"{starting_inputs.shape[0]} sampler for starting_inputs and {ending_inputs.shape[0]} "
            "for endinging_inputs."
        )

        starting_z = self(DatasetOutput(data=starting_inputs)).z
        ending_z = self(DatasetOutput(data=ending_inputs)).z
        t = torch.linspace(0, 1, granularity).to(starting_inputs.device)
        intep_line = (
            torch.kron(
                starting_z.reshape(starting_z.shape[0], -1), (1 - t).unsqueeze(-1)
            )
            + torch.kron(ending_z.reshape(ending_z.shape[0], -1), t.unsqueeze(-1))
        ).reshape((starting_z.shape[0] * t.shape[0],) + (starting_z.shape[1:]))

        decoded_line = self.decoder(intep_line).reconstruction.reshape(
            (
                starting_inputs.shape[0],
                t.shape[0],
            )
            + (starting_inputs.shape[1:])
        )

        return decoded_line

    def update(self):
        """Method that allows model update during the training (at the end of a training epoch)

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """

    def save(self, dir_path: str):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file. If the
        model to save used custom encoder (resp. decoder) provided by the user, these are also
        saved as ``decoder.pkl`` (resp. ``decoder.pkl``).

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

            except FileNotFoundError as e:
                raise e

        env_spec.save_json(dir_path, "environment")
        self.model_config.save_json(dir_path, "model_config")

        # only save .pkl if custom architecture provided
        if not self.model_config.uses_default_encoder:
            with open(os.path.join(dir_path, "encoder.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.encoder))
                cloudpickle.dump(self.encoder, fp)

        if not self.model_config.uses_default_decoder:
            with open(os.path.join(dir_path, "decoder.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.decoder))
                cloudpickle.dump(self.decoder, fp)

        torch.save(model_dict, os.path.join(dir_path, "model.pt"))

    def push_to_hf_hub(self, hf_hub_path: str):  # pragma: no cover
        """Method allowing to save your model directly on the Hugging Face hub.
        You will need to have the `huggingface_hub` package installed and a valid Hugging Face
        account. You can install the package using

        .. code-block:: bash

            python -m pip install huggingface_hub

        end then login using

        .. code-block:: bash

            huggingface-cli login

        Args:
            hf_hub_path (str): path to your repo on the Hugging Face hub.
        """
        if not hf_hub_is_available():
            raise ModuleNotFoundError(
                "`huggingface_hub` package must be installed to push your model to the HF hub. "
                "Run `python -m pip install huggingface_hub` and log in to your account with "
                "`huggingface-cli login`."
            )

        else:
            from huggingface_hub import CommitOperationAdd, HfApi

        logger.info(
            f"Uploading {self.model_name} model to {hf_hub_path} repo in HF hub..."
        )

        tempdir = tempfile.mkdtemp()

        self.save(tempdir)

        model_files = os.listdir(tempdir)

        api = HfApi()
        hf_operations = []

        for file in model_files:
            hf_operations.append(
                CommitOperationAdd(
                    path_in_repo=file,
                    path_or_fileobj=f"{str(os.path.join(tempdir, file))}",
                )
            )

        with open(os.path.join(tempdir, "model_card.md"), "w") as f:
            f.write(model_card_template)

        hf_operations.append(
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=os.path.join(tempdir, "model_card.md"),
            )
        )

        try:
            api.create_commit(
                commit_message=f"Uploading {self.model_name} in {hf_hub_path}",
                repo_id=hf_hub_path,
                operations=hf_operations,
            )
            logger.info(
                f"Successfully uploaded {self.model_name} to {hf_hub_path} repo in HF hub!"
            )

        except:
            from huggingface_hub import create_repo

            repo_name = os.path.basename(os.path.normpath(hf_hub_path))
            logger.info(
                f"Creating {repo_name} in the HF hub since it does not exist..."
            )
            create_repo(repo_id=repo_name)
            logger.info(f"Successfully created {repo_name} in the HF hub!")

            api.create_commit(
                commit_message=f"Uploading {self.model_name} in {hf_hub_path}",
                repo_id=hf_hub_path,
                operations=hf_operations,
            )

        shutil.rmtree(tempdir)

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
    def _load_custom_encoder_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)

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
        cls._check_python_version_from_folder(dir_path=dir_path)

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

    @classmethod
    def load_from_hf_hub(cls, hf_hub_path: str, allow_pickle=False):  # pragma: no cover
        """Class method to be used to load a pretrained model from the Hugging Face hub

        Args:
            hf_hub_path (str): The path where the model should have been be saved on the
                hugginface hub.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        if not hf_hub_is_available():
            raise ModuleNotFoundError(
                "`huggingface_hub` package must be installed to load models from the HF hub. "
                "Run `python -m pip install huggingface_hub` and log in to your account with "
                "`huggingface-cli login`."
            )

        else:
            from huggingface_hub import hf_hub_download

        logger.info(f"Downloading {cls.__name__} files for rebuilding...")

        _ = hf_hub_download(repo_id=hf_hub_path, filename="environment.json")
        config_path = hf_hub_download(repo_id=hf_hub_path, filename="model_config.json")
        dir_path = os.path.dirname(config_path)

        _ = hf_hub_download(repo_id=hf_hub_path, filename="model.pt")

        model_config = cls._load_model_config_from_folder(dir_path)

        if (
            cls.__name__ + "Config" != model_config.name
            and cls.__name__ + "_Config" != model_config.name
        ):
            warnings.warn(
                f"You are trying to load a "
                f"`{ cls.__name__}` while a "
                f"`{model_config.name}` is given."
            )

        model_weights = cls._load_model_weights_from_folder(dir_path)

        if (
            not model_config.uses_default_encoder
            or not model_config.uses_default_decoder
        ) and not allow_pickle:
            warnings.warn(
                "You are about to download pickled files from the HF hub that may have "
                "been created by a third party and so could potentially harm your computer. If you "
                "are sure that you want to download them set `allow_pickle=true`."
            )

        else:

            if not model_config.uses_default_encoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename="encoder.pkl")
                encoder = cls._load_custom_encoder_from_folder(dir_path)

            else:
                encoder = None

            if not model_config.uses_default_decoder:
                _ = hf_hub_download(repo_id=hf_hub_path, filename="decoder.pkl")
                decoder = cls._load_custom_decoder_from_folder(dir_path)

            else:
                decoder = None

            logger.info(f"Successfully downloaded {cls.__name__} model!")

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

    @classmethod
    def _check_python_version_from_folder(cls, dir_path: str):
        if "environment.json" in os.listdir(dir_path):
            env_spec = EnvironmentConfig.from_json_file(
                os.path.join(dir_path, "environment.json")
            )
            python_version = env_spec.python_version
            python_version_minor = python_version.split(".")[1]

            if python_version_minor == "7" and sys.version_info[1] > 7:
                raise LoadError(
                    "Trying to reload a model saved with python3.7 with python3.8+. "
                    "Please create a virtual env with python 3.7 to reload this model."
                )

            elif int(python_version_minor) >= 8 and sys.version_info[1] == 7:
                raise LoadError(
                    "Trying to reload a model saved with python3.8+ with python3.7. "
                    "Please create a virtual env with python 3.8+ to reload this model."
                )
