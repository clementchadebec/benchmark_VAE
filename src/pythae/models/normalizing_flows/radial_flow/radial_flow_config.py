from pydantic.dataclasses import dataclass

from ..base import BaseNFConfig


@dataclass
class RadialFlowConfig(BaseNFConfig):
    """This is the MADE model configuration instance.

    Parameters:
        input_dim (tuple): The input data dimension. Default: None.
        activation (str): The activation function to be applied. Choices: ['linear', 'tanh', 'elu'].
            Default: 'tanh'.
    """
    pass