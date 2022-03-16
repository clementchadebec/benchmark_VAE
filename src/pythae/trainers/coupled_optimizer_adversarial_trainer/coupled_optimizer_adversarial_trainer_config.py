from typing import Union

from pydantic.dataclasses import dataclass

from ..base_trainer import BaseTrainerConfig


@dataclass
class CoupledOptimizerAdversarialTrainerConfig(BaseTrainerConfig):
    """
    CoupledOptimizerAdversarialTrainer config class.

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
        encoder_optim_decay (float): The decay to apply in the optimizer. Default: 0
        decoder_optim_decay (float): The decay to apply in the optimizer. Default: 0
        discriminator_optim_decay (float): The decay to apply in the optimizer. Default: 0
    """
    encoder_optim_decay: float = 0
    decoder_optim_decay: float = 0
    discriminator_optim_decay: float = 0
