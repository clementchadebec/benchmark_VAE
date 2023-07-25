from pydantic.dataclasses import dataclass

from ..base import BaseSamplerConfig


@dataclass
class VAMPSamplerConfig(BaseSamplerConfig):
    """This is the VAMP prior sampler configuration instance deriving from
    :class:`BaseSamplerConfig`.
    """

    pass
