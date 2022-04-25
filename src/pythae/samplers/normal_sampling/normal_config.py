from pydantic.dataclasses import dataclass

from ...samplers import BaseSamplerConfig


@dataclass
class NormalSamplerConfig(BaseSamplerConfig):
    """NormalSampler config class.

    N/A
    """
