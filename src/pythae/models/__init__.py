""" 
This is the heart of pythae! 
Here are implemented some of the most common (Variational) Autoencoders models.

By convention, each implemented model is stored in a folder located in :class:`pythae.models`
and named likewise the model. The following modules can be found in this folder:

- | *modelname_config.py*: Contains a :class:`ModelNameConfig` instance inheriting
    from either :class:`~pythae.models.base.AEConfig` for Autoencoder models or 
    :class:`~pythae.models.base.VAEConfig` for Variational Autoencoder models. 
- | *modelname_model.py*: An implementation of the model inheriting either from
    :class:`~pythae.models.AE` for Autoencoder models or 
    :class:`~pythae.models.base.VAE` for Variational Autoencoder models. 
- *modelname_utils.py* (optional): A module where utils methods are stored.
"""

from .adversarial_ae import Adversarial_AE, Adversarial_AE_Config
from .ae import AE, AEConfig
from .auto_model import AutoModel
from .base import BaseAE, BaseAEConfig
from .beta_tc_vae import BetaTCVAE, BetaTCVAEConfig
from .beta_vae import BetaVAE, BetaVAEConfig
from .ciwae import CIWAE, CIWAEConfig
from .disentangled_beta_vae import DisentangledBetaVAE, DisentangledBetaVAEConfig
from .factor_vae import FactorVAE, FactorVAEConfig
from .hvae import HVAE, HVAEConfig
from .info_vae import INFOVAE_MMD, INFOVAE_MMD_Config
from .iwae import IWAE, IWAEConfig
from .miwae import MIWAE, MIWAEConfig
from .msssim_vae import MSSSIM_VAE, MSSSIM_VAEConfig
from .piwae import PIWAE, PIWAEConfig
from .pvae import PoincareVAE, PoincareVAEConfig
from .rae_gp import RAE_GP, RAE_GP_Config
from .rae_l2 import RAE_L2, RAE_L2_Config
from .rhvae import RHVAE, RHVAEConfig
from .svae import SVAE, SVAEConfig
from .vae import VAE, VAEConfig
from .vae_gan import VAEGAN, VAEGANConfig
from .vae_iaf import VAE_IAF, VAE_IAF_Config
from .vae_lin_nf import VAE_LinNF, VAE_LinNF_Config
from .vamp import VAMP, VAMPConfig
from .vq_vae import VQVAE, VQVAEConfig
from .wae_mmd import WAE_MMD, WAE_MMD_Config

__all__ = [
    "AutoModel",
    "BaseAE",
    "BaseAEConfig",
    "AE",
    "AEConfig",
    "WAE_MMD",
    "WAE_MMD_Config",
    "VAE",
    "VAEConfig",
    "BetaVAE",
    "BetaVAEConfig",
    "VAMP",
    "VAMPConfig",
    "HVAE",
    "HVAEConfig",
    "RHVAE",
    "RHVAEConfig",
    "RAE_L2",
    "RAE_L2_Config",
    "RAE_GP",
    "RAE_GP_Config",
    "IWAE",
    "IWAEConfig",
    "INFOVAE_MMD",
    "INFOVAE_MMD_Config",
    "VQVAE",
    "VQVAEConfig",
    "Adversarial_AE",
    "Adversarial_AE_Config",
    "VAEGAN",
    "VAEGANConfig",
    "MSSSIM_VAE",
    "MSSSIM_VAEConfig",
    "SVAE",
    "SVAEConfig",
    "DisentangledBetaVAE",
    "DisentangledBetaVAEConfig",
    "FactorVAE",
    "FactorVAEConfig",
    "BetaTCVAE",
    "BetaTCVAEConfig",
    "VAE_LinNF",
    "VAE_LinNF_Config",
    "VAE_IAF",
    "VAE_IAF_Config",
    "PoincareVAE",
    "PoincareVAEConfig",
    "CIWAE",
    "CIWAEConfig",
    "MIWAE",
    "MIWAEConfig",
    "PIWAE",
    "PIWAEConfig",
]
