from pydantic.dataclasses import dataclass

from ..base import BaseNFConfig


@dataclass
class PlanarFlowConfig(BaseNFConfig):
    """This is the MADE model configuration instance.

    Parameters:
        input_dim (tuple): The input data dimension. Default: None.
        activation (str): The activation function to be applied. Choices: ['linear', 'tanh', 'elu'].
            Default: 'tanh'.
    """
    activation: str = 'tanh'

    def __post_init_post_parse__(self):
        assert self.activation in ['linear', 'tanh', 'elu'], (
            f"'{f}' doesn't correspond to an activation handled by the model. "
            "Available activations ['linear', 'tanh', 'elu']"
        )
