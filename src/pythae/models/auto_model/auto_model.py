import json
import os

import torch.nn as nn
from ..base.base_utils import hf_hub_is_available

class AutoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def load_from_folder(cls, dir_path: str):
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
            from ..adversarial_ae import Adversarial_AE
            model = Adversarial_AE.load_from_folder(dir_path=dir_path)

        elif model_name == "AEConfig":
            from ..ae import AE
            model = AE.load_from_folder(dir_path=dir_path)

        elif model_name == "BetaTCVAEConfig":
            from ..beta_tc_vae import BetaTCVAE
            model = BetaTCVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "BetaVAEConfig":
            from ..beta_vae import BetaVAE
            model = BetaVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "DisentangledBetaVAEConfig":
            from ..disentangled_beta_vae import DisentangledBetaVAE
            model = DisentangledBetaVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "FactorVAEConfig":
            from ..factor_vae import FactorVAE
            model = FactorVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "HVAEConfig":
            from ..hvae import HVAE
            model = HVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "INFOVAE_MMD_Config":
            from ..info_vae import INFOVAE_MMD
            model = INFOVAE_MMD.load_from_folder(dir_path=dir_path)

        elif model_name == "IWAEConfig":
            from ..iwae import IWAE
            model = IWAE.load_from_folder(dir_path=dir_path)

        elif model_name == "MSSSIM_VAEConfig":
            from ..msssim_vae import MSSSIM_VAE
            model = MSSSIM_VAE.load_from_folder(dir_path=dir_path)

        elif model_name == "RAE_GP_Config":
            from ..rae_gp import RAE_GP
            model = RAE_GP.load_from_folder(dir_path=dir_path)

        elif model_name == "RAE_L2_Config":
            from ..rae_l2 import RAE_L2
            model = RAE_L2.load_from_folder(dir_path=dir_path)

        elif model_name == "RHVAEConfig":
            from ..rhvae import RHVAE
            model = RHVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "SVAEConfig":
            from ..svae import SVAE
            model = SVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "VAEConfig":
            from ..vae import VAE
            model = VAE.load_from_folder(dir_path=dir_path)

        elif model_name == "VAEGANConfig":
            from ..vae_gan import VAEGAN
            model = VAEGAN.load_from_folder(dir_path=dir_path)

        elif model_name == "VAE_IAF_Config":
            from ..vae_iaf import VAE_IAF
            model = VAE_IAF.load_from_folder(dir_path=dir_path)

        elif model_name == "VAE_LinNF_Config":
            from ..vae_lin_nf import VAE_LinNF
            model = VAE_LinNF.load_from_folder(dir_path=dir_path)

        elif model_name == "VAMPConfig":
            from ..vamp import VAMP
            model = VAMP.load_from_folder(dir_path=dir_path)

        elif model_name == "VQVAEConfig":
            from ..vq_vae import VQVAE
            model = VQVAE.load_from_folder(dir_path=dir_path)

        elif model_name == "WAE_MMD_Config":
            from ..wae_mmd import WAE_MMD
            model = WAE_MMD.load_from_folder(dir_path=dir_path)

        elif model_name == "MAFConfig":
            from ..normalizing_flows import MAF
            model = MAF.load_from_folder(dir_path=dir_path)

        elif model_name == "IAFConfig":
            from ..normalizing_flows import IAF
            model = IAF.load_from_folder(dir_path=dir_path)

        elif model_name == "PlanarFlowConfig":
            from ..normalizing_flows import PlanarFlow
            model = PlanarFlow.load_from_folder(dir_path=dir_path)

        elif model_name == "RadialFlowConfig":
            from ..normalizing_flows import RadialFlow
            model = RadialFlow.load_from_folder(dir_path=dir_path)

        elif model_name == "MADEConfig":
            from ..normalizing_flows import MADE
            model = MADE.load_from_folder(dir_path=dir_path)

        elif model_name == "PixelCNNConfig":
            from ..normalizing_flows import PixelCNN
            model = PixelCNN.load_from_folder(dir_path=dir_path)

        else:
            raise NameError(
                "Cannot reload automatically the model... "
                f"The model name in the `model_config.json may be corrupted. Got {model_name}"
            )

        return model


    @classmethod
    def load_from_hf_hub(cls, hf_hub_path: str): # pragma: no cover
        """Class method to be used to load a automaticaly a pretrained model from the huggingface hub

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
                "`huggingface_hub` package must be installed to push you model to the HF hub. "
                "Run `python -m pip install huggingface_hub` and log in to your account with "
                "`huggingface-cli login`."
            )

        else:
            from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=hf_hub_path, filename="model_config.json")
        dir_path = os.path.dirname(config_path)

        with open(os.path.join(dir_path, "model_config.json")) as f:
            model_name = json.load(f)["name"]

        if model_name == "Adversarial_AE_Config":
            from ..adversarial_ae import Adversarial_AE
            model = Adversarial_AE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "AEConfig":
            from ..ae import AE
            model = AE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "BetaTCVAEConfig":
            from ..beta_tc_vae import BetaTCVAE
            model = BetaTCVAE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "BetaVAEConfig":
            from ..beta_vae import BetaVAE
            model = BetaVAE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "DisentangledBetaVAEConfig":
            from ..disentangled_beta_vae import DisentangledBetaVAE
            model = DisentangledBetaVAE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "FactorVAEConfig":
            from ..factor_vae import FactorVAE
            model = FactorVAE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "HVAEConfig":
            from ..hvae import HVAE
            model = HVAE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "INFOVAE_MMD_Config":
            from ..info_vae import INFOVAE_MMD
            model = INFOVAE_MMD.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "IWAEConfig":
            from ..iwae import IWAE
            model = IWAE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "MSSSIM_VAEConfig":
            from ..msssim_vae import MSSSIM_VAE
            model = MSSSIM_VAE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "RAE_GP_Config":
            from ..rae_gp import RAE_GP
            model = RAE_GP.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "RAE_L2_Config":
            from ..rae_l2 import RAE_L2
            model = RAE_L2.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "RHVAEConfig":
            from ..rhvae import RHVAE
            model = RHVAE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "SVAEConfig":
            from ..svae import SVAE
            model = SVAE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "VAEConfig":
            from ..vae import VAE
            model = VAE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "VAEGANConfig":
            from ..vae_gan import VAEGAN
            model = VAEGAN.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "VAE_IAF_Config":
            from ..vae_iaf import VAE_IAF
            model = VAE_IAF.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "VAE_LinNF_Config":
            from ..vae_lin_nf import VAE_LinNF
            model = VAE_LinNF.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "VAMPConfig":
            from ..vamp import VAMP
            model = VAMP.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "VQVAEConfig":
            from ..vq_vae import VQVAE
            model = VQVAE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "WAE_MMD_Config":
            from ..wae_mmd import WAE_MMD
            model = WAE_MMD.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "MAFConfig":
            from ..normalizing_flows import MAF
            model = MAF.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "IAFConfig":
            from ..normalizing_flows import IAF
            model = IAF.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "PlanarFlowConfig":
            from ..normalizing_flows import PlanarFlow
            model = PlanarFlow.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "RadialFlowConfig":
            from ..normalizing_flows import RadialFlow
            model = RadialFlow.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "MADEConfig":
            from ..normalizing_flows import MADE
            model = MADE.load_from_hf_hub(hf_hub_path=hf_hub_path)

        elif model_name == "PixelCNNConfig":
            from ..normalizing_flows import PixelCNN
            model = PixelCNN.load_from_hf_hub(hf_hub_path=hf_hub_path)

        else:
            raise NameError(
                "Cannot reload automatically the model... "
                f"The model name in the `model_config.json may be corrupted. Got {model_name}"
            )

        return model
