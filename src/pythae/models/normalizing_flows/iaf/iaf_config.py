from typing import Union

from pydantic.dataclasses import dataclass

from ..base import BaseNFConfig


@dataclass
class IAFConfig(BaseNFConfig):
    """This is the Inverse Autoregressive Flow model configuration instance.

    Parameters:
        input_dim (tuple): The input data dimension. Default: None.
        n_blocks (int): The number of IAF blocks to consider in the flow. Default: 2.
        n_hidden_in_made (int): The number of hidden layers in the MADE models. Default: 3.
        hidden_size (list): The number of unit in each hidder layer. The same number of units is
            used across the `n_hidden_in_made` and `n_blocks`. Default: 128.
        context_dim (int): The dimension of the context. Default: None.
        include_batch_norm (bool): Whether to include batch normalization after each
            :class:`~pythae.models.normalizing_flows.MADE` layers. Default: False.
    """

    n_blocks: int = 2
    n_hidden_in_made: int = 3
    context_dim: Union[int, None] = None
    hidden_size: int = 128
    include_batch_norm: bool = False
