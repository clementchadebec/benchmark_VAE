from pydantic.dataclasses import dataclass

from pythae.config import BaseConfig


@dataclass
class AutoConfig(BaseConfig):
    @classmethod
    def from_json_file(cls, json_path):
        """Creates a :class:`~pythae.config.BaseAEConfig` instance from a JSON config file. It
        builds automatically the correct config for any `pythae.models`.

        Args:
            json_path (str): The path to the json file containing all the parameters

        Returns:
            :class:`BaseAEConfig`: The created instance
        """

        config_dict = cls._dict_from_json(json_path)
        config_name = config_dict.pop("name")

        if config_name == "BaseAEConfig":
            from ..base import BaseAEConfig

            model_config = BaseAEConfig.from_json_file(json_path)

        elif config_name == "Adversarial_AE_Config":
            from ..adversarial_ae import Adversarial_AE_Config

            model_config = Adversarial_AE_Config.from_json_file(json_path)

        elif config_name == "AEConfig":
            from ..ae import AEConfig

            model_config = AEConfig.from_json_file(json_path)

        elif config_name == "BetaTCVAEConfig":
            from ..beta_tc_vae import BetaTCVAEConfig

            model_config = BetaTCVAEConfig.from_json_file(json_path)

        elif config_name == "BetaVAEConfig":
            from ..beta_vae import BetaVAEConfig

            model_config = BetaVAEConfig.from_json_file(json_path)

        elif config_name == "DisentangledBetaVAEConfig":
            from ..disentangled_beta_vae import DisentangledBetaVAEConfig

            model_config = DisentangledBetaVAEConfig.from_json_file(json_path)

        elif config_name == "FactorVAEConfig":
            from ..factor_vae import FactorVAEConfig

            model_config = FactorVAEConfig.from_json_file(json_path)

        elif config_name == "HVAEConfig":
            from ..hvae import HVAEConfig

            model_config = HVAEConfig.from_json_file(json_path)

        elif config_name == "INFOVAE_MMD_Config":
            from ..info_vae import INFOVAE_MMD_Config

            model_config = INFOVAE_MMD_Config.from_json_file(json_path)

        elif config_name == "IWAEConfig":
            from ..iwae import IWAEConfig

            model_config = IWAEConfig.from_json_file(json_path)

        elif config_name == "MSSSIM_VAEConfig":
            from ..msssim_vae import MSSSIM_VAEConfig

            model_config = MSSSIM_VAEConfig.from_json_file(json_path)

        elif config_name == "RAE_GP_Config":
            from ..rae_gp import RAE_GP_Config

            model_config = RAE_GP_Config.from_json_file(json_path)

        elif config_name == "RAE_L2_Config":
            from ..rae_l2 import RAE_L2_Config

            model_config = RAE_L2_Config.from_json_file(json_path)

        elif config_name == "RHVAEConfig":
            from ..rhvae import RHVAEConfig

            model_config = RHVAEConfig.from_json_file(json_path)

        elif config_name == "SVAEConfig":
            from ..svae import SVAEConfig

            model_config = SVAEConfig.from_json_file(json_path)

        elif config_name == "VAEConfig":
            from ..vae import VAEConfig

            model_config = VAEConfig.from_json_file(json_path)

        elif config_name == "VAEGANConfig":
            from ..vae_gan import VAEGANConfig

            model_config = VAEGANConfig.from_json_file(json_path)

        elif config_name == "VAE_IAF_Config":
            from ..vae_iaf import VAE_IAF_Config

            model_config = VAE_IAF_Config.from_json_file(json_path)

        elif config_name == "VAE_LinNF_Config":
            from ..vae_lin_nf import VAE_LinNF_Config

            model_config = VAE_LinNF_Config.from_json_file(json_path)

        elif config_name == "VAMPConfig":
            from ..vamp import VAMPConfig

            model_config = VAMPConfig.from_json_file(json_path)

        elif config_name == "VQVAEConfig":
            from ..vq_vae import VQVAEConfig

            model_config = VQVAEConfig.from_json_file(json_path)

        elif config_name == "WAE_MMD_Config":
            from ..wae_mmd import WAE_MMD_Config

            model_config = WAE_MMD_Config.from_json_file(json_path)

        elif config_name == "MAFConfig":
            from ..normalizing_flows import MAFConfig

            model_config = MAFConfig.from_json_file(json_path)

        elif config_name == "IAFConfig":
            from ..normalizing_flows import IAFConfig

            model_config = IAFConfig.from_json_file(json_path)

        elif config_name == "PlanarFlowConfig":
            from ..normalizing_flows import PlanarFlowConfig

            model_config = PlanarFlowConfig.from_json_file(json_path)

        elif config_name == "RadialFlowConfig":
            from ..normalizing_flows import RadialFlowConfig

            model_config = RadialFlowConfig.from_json_file(json_path)

        elif config_name == "MADEConfig":
            from ..normalizing_flows import MADEConfig

            model_config = MADEConfig.from_json_file(json_path)

        elif config_name == "PixelCNNConfig":
            from ..normalizing_flows import PixelCNNConfig

            model_config = PixelCNNConfig.from_json_file(json_path)

        elif config_name == "PoincareVAEConfig":
            from ..pvae import PoincareVAEConfig

            model_config = PoincareVAEConfig.from_json_file(json_path)

        elif config_name == "CIWAEConfig":
            from ..ciwae import CIWAEConfig

            model_config = CIWAEConfig.from_json_file(json_path)

        elif config_name == "MIWAEConfig":
            from ..miwae import MIWAEConfig

            model_config = MIWAEConfig.from_json_file(json_path)

        elif config_name == "PIWAEConfig":
            from ..piwae import PIWAEConfig

            model_config = PIWAEConfig.from_json_file(json_path)

        else:
            raise NameError(
                "Cannot reload automatically the model configuration... "
                f"The model name in the `model_config.json may be corrupted. Got `{config_name}`"
            )

        return model_config
