from pydantic.dataclasses import dataclass

from ...samplers import BaseSamplerConfig


@dataclass
class GaussianMixtureSamplerConfig(BaseSamplerConfig):
    """This is the Gaussian mixture sampler configuration instance deriving from
    :class:`BaseSamplerConfig`.

    Parameters:
        n_components (int): The number of Gaussians in the mixture
    """

    n_components: int = 10
