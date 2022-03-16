from typing import Union

from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class BaseTrainerConfig(BaseConfig):
    """
    BaseTrainer config class stating the main training arguments.

    Parameters:

        output_dir (str): The directory where model checkpoints, configs and final
            model will be stored. Default: None.

        batch_size (int): The number of training samples per batch. Default 100
        num_epochs (int): The maximal number of epochs for training. Default: 100
        learning_rate (int): The learning rate applied to the `Optimizer`. Default: 1e-4
        steps_saving (int): A model checkpoint will be saved every `steps_saving` epoch. 
            Default: None
        keep_best_on_train (bool): Whether to keep the best model on the train set. Default: False.
        seed (int): The random seed for reproducibility
        no_cuda (bool): Disable `cuda` training. Default: False
    """

    output_dir: str = None
    batch_size: int = 100
    num_epochs: int = 100
    learning_rate: float = 1e-4
    steps_saving: Union[int, None] = None
    keep_best_on_train: bool = False
    seed: int = 8
    no_cuda: bool = False
