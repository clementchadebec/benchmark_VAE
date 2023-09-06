from typing import Union

import torch.nn as nn
from pydantic.dataclasses import dataclass

from ..base_trainer import BaseTrainerConfig


@dataclass
class CoupledOptimizerAdversarialTrainerConfig(BaseTrainerConfig):
    """
    CoupledOptimizerAdversarialTrainer config class.

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
        encoder_optimizer_cls (str): The name of the `torch.optim.Optimizer` used for
            the training of the encoder. Default: :class:`~torch.optim.Adam`.
        encoder_optimizer_params (dict): A dict containing the parameters to use for the
            `torch.optim.Optimizer` for the encoder. If None, uses the default parameters.
            Default: None.
        encoder_scheduler_cls (str): The name of the `torch.optim.lr_scheduler` used for
            the training of the encoder. Default :class:`~torch.optim.Adam`.
        encoder_scheduler_params (dict): A dict containing the parameters to use for the
            `torch.optim.le_scheduler`  for the encoder. If None, uses the default parameters.
            Default: None.
        decoder_optimizer_cls (str): The name of the `torch.optim.Optimizer` used for
            the training of the decoder. Default: :class:`~torch.optim.Adam`.
        decoder_optimizer_params (dict): A dict containing the parameters to use for the
            `torch.optim.Optimizer` for the decoder. If None, uses the default parameters.
            Default: None.
        decoder_scheduler_cls (str): The name of the `torch.optim.lr_scheduler` used for
            the training of the decoder. Default :class:`~torch.optim.Adam`.
        decoder_scheduler_params (dict): A dict containing the parameters to use for the
            `torch.optim.le_scheduler`  for the decoder. If None, uses the default parameters.
            Default: None.
        discriminator_optimizer_cls (str): The name of the `torch.optim.Optimizer` used for
            the training of the discriminator. Default: :class:`~torch.optim.Adam`.
        discriminator_optimizer_params (dict): A dict containing the parameters to use for the
            `torch.optim.Optimizer` for the discriminator. If None, uses the default parameters.
            Default: None.
        discriminator_scheduler_cls (str): The name of the `torch.optim.lr_scheduler` used for
            the training of the discriminator. Default :class:`~torch.optim.Adam`.
        discriminator_scheduler_params (dict): A dict containing the parameters to use for the
            `torch.optim.le_scheduler`  for the discriminator. If None, uses the default parameters.
            Default: None.
        encoder_learning_rate (int): The learning rate applied to the `Optimizer` for the encoder.
            Default: 1e-4
        decoder_learning_rate (int): The learning rate applied to the `Optimizer` for the encoder.
            Default: 1e-4
        discriminator_learning_rate (int): The learning rate applied to the `Optimizer` for the
            discriminator. Default: 1e-4
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
        master_addr (str): The master address for distributed training. Default: 'localhost'
        master_port (str): The master port for distributed training. Default: '12345'"""

    encoder_optimizer_cls: str = "Adam"
    encoder_optimizer_params: Union[dict, None] = None
    encoder_scheduler_cls: str = None
    encoder_scheduler_params: Union[dict, None] = None
    discriminator_optimizer_cls: str = "Adam"
    decoder_optimizer_cls: str = "Adam"
    decoder_optimizer_params: Union[dict, None] = None
    decoder_scheduler_cls: str = None
    decoder_scheduler_params: Union[dict, None] = None
    discriminator_optimizer_cls: str = "Adam"
    discriminator_optimizer_params: Union[dict, None] = None
    discriminator_scheduler_cls: str = None
    discriminator_scheduler_params: Union[dict, None] = None
    encoder_learning_rate: float = 1e-4
    decoder_learning_rate: float = 1e-4
    discriminator_learning_rate: float = 1e-4

    def __post_init__(self):
        """Check compatibilty"""
        super().__post_init__()

        # Encoder optimizer and scheduler
        try:
            import torch.optim as optim

            encoder_optimizer_cls = getattr(optim, self.encoder_optimizer_cls)
        except AttributeError as e:
            raise AttributeError(
                f"Unable to import `{self.encoder_optimizer_cls}` encoder optimizer "
                "from 'torch.optim'. Check spelling and that it is part of "
                "'torch.optim.Optimizers.'"
            )
        if self.encoder_optimizer_params is not None:
            try:
                encoder_optimizer = encoder_optimizer_cls(
                    nn.Linear(2, 2).parameters(),
                    lr=self.encoder_learning_rate,
                    **self.encoder_optimizer_params,
                )
            except TypeError as e:
                raise TypeError(
                    "Error in optimizer's parameters. Check that the provided dict contains only "
                    f"keys and values suitable for `{encoder_optimizer_cls}` optimizer. "
                    f"Got {self.encoder_optimizer_params} as parameters.\n"
                    f"Exception raised: {type(e)} with message: " + str(e)
                ) from e
        else:
            encoder_optimizer = encoder_optimizer_cls(
                nn.Linear(2, 2).parameters(),
                lr=self.encoder_learning_rate,
            )

        if self.encoder_scheduler_cls is not None:
            try:
                import torch.optim.lr_scheduler as schedulers

                encoder_scheduder_cls = getattr(schedulers, self.encoder_scheduler_cls)
            except AttributeError as e:
                raise AttributeError(
                    f"Unable to import `{self.encoder_scheduler_cls}` encoder scheduler from "
                    "'torch.optim.lr_scheduler'. Check spelling and that it is part of "
                    "'torch.optim.lr_scheduler.'"
                )

            if self.encoder_scheduler_params is not None:
                try:
                    encoder_scheduder_cls(
                        encoder_optimizer, **self.encoder_scheduler_params
                    )
                except TypeError as e:
                    raise TypeError(
                        "Error in scheduler's parameters. Check that the provided dict contains only "
                        f"keys and values suitable for `{encoder_scheduder_cls}` scheduler. "
                        f"Got {self.encoder_scheduler_params} as parameters.\n"
                        f"Exception raised: {type(e)} with message: " + str(e)
                    ) from e

        # Decoder optimizer and scheduler
        try:
            import torch.optim as optim

            decoder_optimizer_cls = getattr(optim, self.decoder_optimizer_cls)
        except AttributeError as e:
            raise AttributeError(
                f"Unable to import `{self.decoder_optimizer_cls}` decoder optimizer "
                "from 'torch.optim'. Check spelling and that it is part of "
                "'torch.optim.Optimizers.'"
            )
        if self.decoder_optimizer_params is not None:
            try:
                decoder_optimizer = decoder_optimizer_cls(
                    nn.Linear(2, 2).parameters(),
                    lr=self.decoder_learning_rate,
                    **self.decoder_optimizer_params,
                )
            except TypeError as e:
                raise TypeError(
                    "Error in optimizer's parameters. Check that the provided dict contains only "
                    f"keys and values suitable for `{decoder_optimizer_cls}` optimizer. "
                    f"Got {self.decoder_optimizer_params} as parameters.\n"
                    f"Exception raised: {type(e)} with message: " + str(e)
                ) from e
        else:
            decoder_optimizer = decoder_optimizer_cls(
                nn.Linear(2, 2).parameters(), lr=self.decoder_learning_rate
            )

        if self.decoder_scheduler_cls is not None:
            try:
                import torch.optim.lr_scheduler as schedulers

                decoder_scheduder_cls = getattr(schedulers, self.decoder_scheduler_cls)
            except AttributeError as e:
                raise AttributeError(
                    f"Unable to import `{self.decoder_scheduler_cls}` decoder scheduler from "
                    "'torch.optim.lr_scheduler'. Check spelling and that it is part of "
                    "'torch.optim.lr_scheduler.'"
                )

            if self.decoder_scheduler_params is not None:
                try:
                    decoder_scheduder_cls(
                        decoder_optimizer, **self.decoder_scheduler_params
                    )
                except TypeError as e:
                    raise TypeError(
                        "Error in scheduler's parameters. Check that the provided dict contains only "
                        f"keys and values suitable for `{decoder_scheduder_cls}` scheduler. "
                        f"Got {self.decoder_scheduler_params} as parameters.\n"
                        f"Exception raised: {type(e)} with message: " + str(e)
                    ) from e

        # Discriminator optimizer and scheduler
        try:
            discriminator_optimizer_cls = getattr(
                optim, self.discriminator_optimizer_cls
            )
        except AttributeError as e:
            raise AttributeError(
                f"Unable to import `{self.discriminator_optimizer_cls}` discriminator optimizer "
                "from 'torch.optim'. Check spelling and that it is part of "
                "'torch.optim.Optimizers.'"
            )
        if self.discriminator_optimizer_params is not None:
            try:
                discriminator_optimizer = discriminator_optimizer_cls(
                    nn.Linear(2, 2).parameters(),
                    lr=self.discriminator_learning_rate,
                    **self.discriminator_optimizer_params,
                )
            except TypeError as e:
                raise TypeError(
                    "Error in optimizer's parameters. Check that the provided dict contains only "
                    f"keys and values suitable for `{discriminator_optimizer_cls}` optimizer. "
                    f"Got {self.discriminator_optimizer_params} as parameters.\n"
                    f"Exception raised: {type(e)} with message: " + str(e)
                ) from e
        else:
            discriminator_optimizer = discriminator_optimizer_cls(
                nn.Linear(2, 2).parameters(), lr=self.discriminator_learning_rate
            )

        if self.discriminator_scheduler_cls is not None:
            try:
                import torch.optim.lr_scheduler as schedulers

                discriminator_scheduder_cls = getattr(
                    schedulers, self.discriminator_scheduler_cls
                )
            except AttributeError as e:
                raise AttributeError(
                    f"Unable to import `{self.discriminator_scheduler_cls}` discriminator scheduler from "
                    "'torch.optim.lr_scheduler'. Check spelling and that it is part of "
                    "'torch.optim.lr_scheduler.'"
                )

            if self.discriminator_scheduler_params is not None:
                try:
                    discriminator_scheduder_cls(
                        discriminator_optimizer, **self.discriminator_scheduler_params
                    )
                except TypeError as e:
                    raise TypeError(
                        "Error in scheduler's parameters. Check that the provided dict contains only "
                        f"keys and values suitable for `{discriminator_scheduder_cls}` scheduler. "
                        f"Got {self.discriminator_scheduler_params} as parameters.\n"
                        f"Exception raised: {type(e)} with message: " + str(e)
                    ) from e
