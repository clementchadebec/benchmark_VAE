from pydantic.dataclasses import dataclass

from ..base import BaseSamplerConfig


@dataclass
class PoincareDiskSamplerConfig(BaseSamplerConfig):
    """This is the Poincare Disk prior sampler configuration instance deriving from
    :class:`BaseSamplerConfig`.
    """

    pass
