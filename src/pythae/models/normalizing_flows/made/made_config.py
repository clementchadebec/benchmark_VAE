from dataclasses import field
from typing import List, Tuple, Union

from pydantic import validator
from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..base import BaseNFConfig


@dataclass
class MADEConfig(BaseNFConfig):
    """This is the MADE model configuration instance.

    Parameters:
        input_dim (tuple): The input data dimension. Default: None.
        output_dim (tuple): The output data dimension. Default: None.
        hidden_sizes (list): The list of the number of hidden units in the Autoencoder.
            Default: [128].
        degrees_ordering (str): The ordering to use for the mask creation. Can be either
            `sequential` or `random`. Default: `sequential`.
    """

    output_dim: Union[Tuple[int, ...], None] = None
    hidden_sizes: List[int] = field(default_factory=lambda: [128])
    degrees_ordering: Literal["sequential", "random"] = "sequential"
