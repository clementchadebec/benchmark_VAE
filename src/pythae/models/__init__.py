""" Here will be implemented the main Autoencoders architecture

By convention, each implemented model is contained within a folder located in :ref:`pythae.models`
in which are located 4 modules:

- | *model_config.py*: Contains a :class:`OtherModelConfig` instance inheriting
    from :class:`~pythae.models.base.BaseAEConfig` where the model configuration is stored and
    a :class:`OtherModelSamplerConfig` instance inheriting from
    :class:`~pythae.models.base.BaseSamplerConfig` where the configuration of the sampler used
    to generate new samples is defined.
- | *other_model_model.py*: An implementation of the other_model inheriting from
    :class:`~pythae.models.BaseAE`.
- | *other_model_sampler.py*: An implementation of the sampler(s) to use to generate new data
    inheriting from :class:`~pythae.models.base.base_sampler.BaseSampler`.
- *other_model_utils.py*: A module where utils methods are stored.
"""

from .base import BaseAE, BaseAEConfig
from .ae import AE, AEConfig
from .wae_mmd import WAE_MMD, WAE_MMD_Config
from .vae import VAE, VAEConfig
from .beta_vae import BetaVAE, BetaVAEConfig
from .vamp import VAMP, VAMPConfig
from .hvae import HVAE, HVAEConfig
from .rhvae import RHVAE, RHVAEConfig
from .rae_l2 import RAE_L2, RAE_L2_Config


__all__ = [
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
    "RAE_L2_Config"
]
