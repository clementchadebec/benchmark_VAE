from dataclasses import field
from typing import List, Tuple, Union

from pydantic import validator
from pydantic.dataclasses import dataclass

from ..base import BaseNFConfig


@dataclass
class MAFConfig(BaseNFConfig):
    """This is the MADE model configuration instance.

    Parameters:
        input_dim (tuple): The input data dimension. Default: None.
        output_dim (tuple): The output data dimension. Default: None.
        n_made_blocks (int): The number of MADE model to consider in the MAF. Default: 2.
        n_hidden_in_made (int): The number of hidden layers in the MADE models. Default: 3.
        hidden_size (list): The number of unit in each hidder layer. The same number of units is 
            used across the `n_hidden_in_made` and `n_made_blocks`
    """
    n_made_blocks: int = 2
    n_hidden_in_made: int = 3
    hidden_size: int = 128
    include_batch_norm: bool = False