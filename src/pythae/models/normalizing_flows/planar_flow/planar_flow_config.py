from pydantic.dataclasses import dataclass

from ..base import BaseNFConfig


@dataclass
class PlanarFlowConfig(BaseNFConfig):
    """This is the PlanarFlow model configuration instance.

    Parameters:
        input_dim (tuple): The input data dimension. Default: None.
        activation (str): The activation function to be applied. Choices: ['linear', 'tanh', 'elu'].
            Default: 'tanh'.
    """

    activation: str = "tanh"

    def __post_init__(self):
        super().__post_init__()
        assert self.activation in ["linear", "tanh", "elu"], (
            f"'{self.activation}' doesn't correspond to an activation handled by the model. "
            "Available activations ['linear', 'tanh', 'elu']"
        )
