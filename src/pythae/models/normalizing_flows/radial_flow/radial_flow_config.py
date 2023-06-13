from pydantic.dataclasses import dataclass

from ..base import BaseNFConfig


@dataclass
class RadialFlowConfig(BaseNFConfig):
    """This is the RadialFlow model configuration instance.

    Parameters:
        input_dim (tuple): The input data dimension. Default: None.
    """
