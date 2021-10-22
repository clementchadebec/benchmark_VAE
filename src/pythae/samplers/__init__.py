""" Here will be implemented the main samplers used in the :ref:`pythae.models`. 

By convention, each implemented model is contained within a folder located in :ref:`pythae.models`
in which are located 4 modules:

- | *sampler_name_config.py*: Contains a :class:`SamplerNameConfig` instance inheriting
    from :class:`~pythae.samplers.base.base_config.BaseSamplerConfig` where the sampler 
    configuration is stored and 
- | *sampler_name_sampler.py*: An implementation of the sampler_name inheriting from
    :class:`~pythae.samplers.BaseSampler`.
"""

from .base import BaseSampler, BaseSamplerConfig
from .normal_sampling import NormalSampler, NormalSamplerConfig
from .gaussian_mixure import GaussianMixtureSampler, GaussianMixtureSamplerConfig
from .vamp_sampler import VAMPSampler, VAMPSamplerConfig
from .manifold_sampler import RHVAESampler, RHVAESamplerConfig
from .two_stage_vae_sampler import TwoStageVAESampler, TwoStageVAESamplerConfig

__all__ = [
    "BaseSampler",
    "BaseSamplerConfig",
    "NormalSampler",
    "NormalSamplerConfig",
    "GaussianMixtureSampler",
    "GaussianMixtureSamplerConfig",
    "VAMPSampler",
    "VAMPSamplerConfig",
    "RHVAESampler",
    "RHVAESamplerConfig",
    "TwoStageVAESampler",
    "TwoStageVAESamplerConfig"
]
