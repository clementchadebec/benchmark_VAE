from pydantic.dataclasses import dataclass

from ..base import BaseSamplerConfig


@dataclass
class GaussianMixtureSamplerConfig(BaseSamplerConfig):
    """Gaussian mixture sampler config class.

    Parameters:
        n_components (int): The number of Gaussians in the mixture
    """

    n_components: int = 10
