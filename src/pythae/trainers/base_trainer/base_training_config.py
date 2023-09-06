import os
from dataclasses import field
from typing import Union

import torch.nn as nn
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
        train_dataloader_num_workers (int): Number of subprocesses to use for train data loading.
            0 means that the data will be loaded in the main process. Default: 0
        eval_dataloader_num_workers (int): Number of subprocesses to use for evaluation data
            loading. 0 means that the data will be loaded in the main process. Default: 0
        optimizer_cls (str): The name of the `torch.optim.Optimizer` used for
            training. Default: :class:`~torch.optim.Adam`.
        optimizer_params (dict): A dict containing the parameters to use for the
            `torch.optim.Optimizer`. If None, uses the default parameters. Default: None.
        scheduler_cls (str): The name of the `torch.optim.lr_scheduler` used for
            training. If None, no scheduler is used. Default None.
        scheduler_params (dict): A dict containing the parameters to use for the
            `torch.optim.le_scheduler`. If None, uses the default parameters. Default: None.
        learning_rate (int): The learning rate applied to the `Optimizer`. Default: 1e-4
        steps_saving (int): A model checkpoint will be saved every `steps_saving` epoch.
            Default: None
        steps_predict (int): A prediction using the best model will be run every `steps_predict`
            epoch. Default: None
        keep_best_on_train (bool): Whether to keep the best model on the train set. Default: False
        seed (int): The random seed for reproducibility
        no_cuda (bool): Disable `cuda` training. Default: False
        world_size (int): The total number of process to run. Default: -1
        local_rank (int): The rank of the node for distributed training. Default: -1
        rank (int): The rank of the process for distributed training. Default: -1
        dist_backend (str): The distributed backend to use. Default: 'nccl'
        master_addr (str): The master address for distributed training. Default: 'localhost'
        master_port (str): The master port for distributed training. Default: '12345'
        amp (bool): Whether to use auto mixed precision in training. Default: False
    """

    output_dir: Union[str, None] = None
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    num_epochs: int = 100
    train_dataloader_num_workers: int = 0
    eval_dataloader_num_workers: int = 0
    optimizer_cls: str = "Adam"
    optimizer_params: Union[dict, None] = None
    scheduler_cls: Union[str, None] = None
    scheduler_params: Union[dict, None] = None
    learning_rate: float = 1e-4
    steps_saving: Union[int, None] = None
    steps_predict: Union[int, None] = None
    keep_best_on_train: bool = False
    seed: int = 8
    no_cuda: bool = False
    world_size: int = field(default=-1)
    local_rank: int = field(default=-1)
    rank: int = field(default=-1)
    dist_backend: str = field(default="nccl")
    master_addr: str = field(default="localhost")
    master_port: str = field(default="12345")
    amp: bool = False

    def __post_init__(self):
        """Check compatibility and sets up distributed training"""
        super().__post_init__()
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
        if self.master_port == "12345" and env_master_port != "12345":
            self.master_port = env_master_port
        os.environ["MASTER_PORT"] = self.master_port

        try:
            import torch.optim as optim

            optimizer_cls = getattr(optim, self.optimizer_cls)
        except AttributeError as e:
            raise AttributeError(
                f"Unable to import `{self.optimizer_cls}` optimizer from 'torch.optim'. "
                "Check spelling and that it is part of 'torch.optim.Optimizers.'"
            )
        if self.optimizer_params is not None:
            try:
                optimizer = optimizer_cls(
                    nn.Linear(2, 2).parameters(),
                    lr=self.learning_rate,
                    **self.optimizer_params,
                )
            except TypeError as e:
                raise TypeError(
                    "Error in optimizer's parameters. Check that the provided dict contains only "
                    f"keys and values suitable for `{optimizer_cls}` optimizer. "
                    f"Got {self.optimizer_params} as parameters.\n"
                    f"Exception raised: {type(e)} with message: " + str(e)
                ) from e
        else:
            optimizer = optimizer_cls(
                nn.Linear(2, 2).parameters(), lr=self.learning_rate
            )

        if self.scheduler_cls is not None:
            try:
                import torch.optim.lr_scheduler as schedulers

                scheduder_cls = getattr(schedulers, self.scheduler_cls)
            except AttributeError as e:
                raise AttributeError(
                    f"Unable to import `{self.scheduler_cls}` scheduler from "
                    "'torch.optim.lr_scheduler'. Check spelling and that it is part of "
                    "'torch.optim.lr_scheduler.'"
                )

            if self.scheduler_params is not None:
                try:
                    scheduder_cls(optimizer, **self.scheduler_params)
                except TypeError as e:
                    raise TypeError(
                        "Error in scheduler's parameters. Check that the provided dict contains only "
                        f"keys and values suitable for `{scheduder_cls}` scheduler. "
                        f"Got {self.scheduler_params} as parameters.\n"
                        f"Exception raised: {type(e)} with message: " + str(e)
                    ) from e

        if self.no_cuda:
            self.amp = False
