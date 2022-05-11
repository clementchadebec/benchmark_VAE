import json
import os

import torch.nn as nn

from ..adversarial_ae import Adversarial_AE
from ..ae import AE
from ..beta_tc_vae import BetaTCVAE
from ..beta_vae import BetaVAE
from ..disentangled_beta_vae import DisentangledBetaVAE
from ..factor_vae import FactorVAE
from ..hvae import HVAE
from ..info_vae import INFOVAE_MMD
from ..iwae import IWAE
from ..msssim_vae import MSSSIM_VAE
from ..rae_gp import RAE_GP
from ..rae_l2 import RAE_L2
from ..rhvae import RHVAE
from ..svae import SVAE
from ..vae import VAE
from ..vae_gan import VAEGAN
from ..vae_iaf import VAE_IAF
from ..vae_lin_nf import VAE_LinNF
from ..vamp import VAMP
from ..vq_vae import VQVAE
from ..wae_mmd import WAE_MMD


class AutoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

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

        with open(os.path.join(dir_path, "model_config.json")) as f:
            model_name = json.load(f)["name"]

        if model_name == "Adversarial_AE_Config":
            model = Adversarial_AE.load_from_folder(dir_path=dir_path)

        elif model_name == "AEConfig":
            model = AE.load_from_folder(dir_path=dir_path)

        elif model_name == "BetaTCVAEConfig":
            model = BetaTCVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "BetaVAEConfig":
            model = BetaVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "DisentangledBetaVAEConfig":
            model = DisentangledBetaVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "FactorVAEConfig":
            model = FactorVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "HVAEConfig":
            model = HVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "INFOVAE_MMD_Config":
            model = INFOVAE_MMD.load_from_folder(dir_path=dir_path)

        elif model_name == "IWAEConfig":
            model = IWAE.load_from_folder(dir_path=dir_path)

        elif model_name == "MSSSIM_VAEConfig":
            model = MSSSIM_VAE.load_from_folder(dir_path=dir_path)

        elif model_name == "RAE_GP_Config":
            model = RAE_GP.load_from_folder(dir_path=dir_path)

        elif model_name == "RAE_L2_Config":
            model = RAE_L2.load_from_folder(dir_path=dir_path)

        elif model_name == "RHVAEConfig":
            model = RHVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "SVAEConfig":
            model = SVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "VAEConfig":
            model = VAE.load_from_folder(dir_path=dir_path)

        elif model_name == "VAEGANConfig":
            model = VAEGAN.load_from_folder(dir_path=dir_path)

        elif model_name == "VAE_IAF_Config":
            model = VAE_IAF.load_from_folder(dir_path=dir_path)

        elif model_name == "VAE_LinNF_Config":
            model = VAE_LinNF.load_from_folder(dir_path=dir_path)

        elif model_name == "VAMPConfig":
            model = VAMP.load_from_folder(dir_path=dir_path)

        elif model_name == "VQVAEConfig":
            model = VQVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "WAE_MMD_Config":
            model = WAE_MMD.load_from_folder(dir_path=dir_path)

        else:
            raise NameError(
                "Cannot reload automatically the model... "
                f"The model name in the `model_config.json may be corrupted. Got {model_name}"
            )

        return model
