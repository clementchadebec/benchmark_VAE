"""
**Abstract class**

This is the base Sampler architecture module from which all future samplers should inherit.
"""

from .base_sampler import BaseSampler
from .base_sampler_config import BaseSamplerConfig

__all__ = ["BaseSampler", "BaseSamplerConfig"]
