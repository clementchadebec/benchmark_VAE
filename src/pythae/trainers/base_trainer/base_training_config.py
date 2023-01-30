import os
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
        per_device_train_batch_size (int): The number of training samples per batch and per device.
            Default 64
        per_device_eval_batch_size (int): The number of evaluation samples per batch and per device.
            Default 64
        num_epochs (int): The maximal number of epochs for training. Default: 100
        learning_rate (int): The learning rate applied to the `Optimizer`. Default: 1e-4
        steps_saving (int): A model checkpoint will be saved every `steps_saving` epoch.
            Default: None
        steps_saving (int): A prediction using the best model will be run every `steps_predict`
            epoch. Default: None
        keep_best_on_train (bool): Whether to keep the best model on the train set. Default: False
        seed (int): The random seed for reproducibility
        no_cuda (bool): Disable `cuda` training. Default: False
        world_size (int): The total number of process to run. Default: -1
        local_rank (int): The rank of the node for distributed training. Default: -1
        rank (int): The rank of the process for distributed training. Default: -1
        dist_backend (str): The distributed backend to use. Default: 'nccl'

    """

    output_dir: str = None
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    steps_saving: Union[int, None] = None
    steps_predict: Union[int, None] = None
    keep_best_on_train: bool = False
    seed: int = 8
    no_cuda: bool = False
    world_size: int = -1
    local_rank: int = -1
    rank: int = -1
    dist_backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "12345"

    def __post_init_post_parse__(self):
        """Handles environ variables"""
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if self.local_rank == -1 and env_local_rank != -1:
            self.local_rank = env_local_rank

        env_world_size = int(os.environ.get("WORLD_SIZE", -1))
        if self.world_size == -1 and env_world_size != -1:
            self.world_size = env_world_size

        env_rank = int(os.environ.get("RANK", -1))
        if self.rank == -1 and env_rank != -1:
            self.rank = env_rank

        env_master_addr = os.environ.get("MASTER_ADDR", "localhost")
        if self.master_addr == "localhost" and env_master_addr != "localhost":
            self.master_addr = env_master_addr
        os.environ["MASTER_ADDR"] = self.master_addr

        env_master_port = os.environ.get("MASTER_PORT", "12345")
        if env_master_port == "12345" and env_master_port != "12345":
            self.master_port = env_master_port
        os.environ["MASTER_PORT"] = self.master_port
