from pydantic.dataclasses import dataclass

from ...samplers import BaseSamplerConfig

@dataclass
class NormalSampler_Config(BaseSamplerConfig):
    """This is the Normal smapler configuration instance deriving from
    :class:`BaseSamplerConfig`.
    """
    pass