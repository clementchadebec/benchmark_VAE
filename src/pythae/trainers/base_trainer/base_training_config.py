from typing import Union

from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class BaseTrainingConfig(BaseConfig):
    """
    :class:`~pythae.trainers.training_config.TrainingConfig` is the class in which all the training arguments
    are stored.
    This instance is then provided to a :class:`~pythae.trainers.Trainer` instance which performs
    a model training.

    Parameters:

        output_dir (str): The directory where model checkpoints, configs and final
            model will be stored. Default: None.

        batch_size (int): The number of training samples per batch. Default 50
        num_epochs (int): The maximal number of epochs for training. Default: 10000
        learning_rate (int): The learning rate applied to the `Optimizer`. Default: 1e-3
        steps_saving (int): A model checkpoint will be saved every `steps_saving` epoch
        keep_best_on_train (bool): Whether to keep the best model on the train set. Default: False.
        seed (int): The random seed for reprodicibility
        no_cuda (bool): Disable `cuda` training. Default: False
    """

    output_dir: str = None
    batch_size: int = 50
    num_epochs: int = 100
    learning_rate: float = 1e-4
    steps_saving: Union[int, None] = None
    keep_best_on_train: bool = False
    seed: int = 8
    no_cuda: bool = False
