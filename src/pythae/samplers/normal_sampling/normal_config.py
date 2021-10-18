from pydantic.dataclasses import dataclass

from ...samplers import BaseSamplerConfig


@dataclass
class NormalSamplerConfig(BaseSamplerConfig):
    """This is the Normal sampler configuration instance deriving from
    :class:`BaseSamplerConfig`.
    """

    pass
