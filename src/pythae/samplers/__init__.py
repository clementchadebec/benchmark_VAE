""" Here are implemented the main samplers used in the :class:`pythae.models`. 

By convention, each implemented model is contained within a folder located in 
:class:`pythae.samplers` and named likewise the sampler. The following modules can be found in 
this folder:

- | *samplername_config.py*: Contains a :class:`SamplerNameConfig` instance inheriting
    from :class:`~pythae.samplers.base.base_config.BaseSamplerConfig` where the sampler 
    configuration is stored and 
- | *samplername_sampler.py*: An implementation of the sampler_name inheriting from
    :class:`~pythae.samplers.BaseSampler`.
- *samplername_utils.py* (optional): A module where utils methods are stored.
"""

from .base import BaseSampler, BaseSamplerConfig
from .gaussian_mixture import GaussianMixtureSampler, GaussianMixtureSamplerConfig
from .hypersphere_uniform_sampler import (
    HypersphereUniformSampler,
    HypersphereUniformSamplerConfig,
)
from .iaf_sampler import IAFSampler, IAFSamplerConfig
from .maf_sampler import MAFSampler, MAFSamplerConfig
from .manifold_sampler import RHVAESampler, RHVAESamplerConfig
from .normal_sampling import NormalSampler, NormalSamplerConfig
from .pixelcnn_sampler import PixelCNNSampler, PixelCNNSamplerConfig
from .pvae_sampler import PoincareDiskSampler, PoincareDiskSamplerConfig
from .two_stage_vae_sampler import TwoStageVAESampler, TwoStageVAESamplerConfig
from .vamp_sampler import VAMPSampler, VAMPSamplerConfig

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
    "TwoStageVAESamplerConfig",
    "HypersphereUniformSampler",
    "HypersphereUniformSamplerConfig",
    "MAFSampler",
    "MAFSamplerConfig",
    "IAFSampler",
    "IAFSamplerConfig",
    "PixelCNNSampler",
    "PixelCNNSamplerConfig",
    "PoincareDiskSampler",
    "PoincareDiskSamplerConfig",
]
