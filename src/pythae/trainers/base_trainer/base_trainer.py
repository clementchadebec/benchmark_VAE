import contextlib
import datetime
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ...customexception import ModelError
from ...data.datasets import BaseDataset, collate_dataset_output
from ...models import BaseAE
from ..trainer_utils import set_seed
from ..training_callbacks import (
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    TrainingCallback,
)
from .base_training_config import BaseTrainerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseTrainer:
    """Base class to perform model training.

    Args:
        model (BaseAE): A instance of :class:`~pythae.models.BaseAE` to train

        train_dataset (BaseDataset): The training dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        eval_dataset (BaseDataset): The evaluation dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        training_config (BaseTrainerConfig): The training arguments summarizing the main
            parameters used for training. If None, a basic training instance of
            :class:`BaseTrainerConfig` is used. Default: None.

        callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
            A list of callbacks to use during training.
    """

    def __init__(
        self,
        model: BaseAE,
        train_dataset: Union[BaseDataset, DataLoader],
        eval_dataset: Optional[Union[BaseDataset, DataLoader]] = None,
        training_config: Optional[BaseTrainerConfig] = None,
        callbacks: List[TrainingCallback] = None,
    ):

        if training_config is None:
            training_config = BaseTrainerConfig()

        if training_config.output_dir is None:
            output_dir = "dummy_output_dir"
            training_config.output_dir = output_dir

        self.training_config = training_config
        self.model_config = model.model_config
        self.model_name = model.model_name

        # for distributed training
        self.world_size = self.training_config.world_size
        self.local_rank = self.training_config.local_rank
        self.rank = self.training_config.rank
        self.dist_backend = self.training_config.dist_backend

        if self.world_size > 1:
            self.distributed = True
        else:
            self.distributed = False

        if self.distributed:
            device = self._setup_devices()

        else:
            device = (
                "cuda"
                if torch.cuda.is_available() and not self.training_config.no_cuda
                else "cpu"
            )

        self.amp_context = (
            torch.autocast("cuda")
            if self.training_config.amp
            else contextlib.nullcontext()
        )

        if (
            hasattr(model.model_config, "reconstruction_loss")
            and model.model_config.reconstruction_loss == "bce"
        ):
            self.amp_context = contextlib.nullcontext()

        self.device = device

        # place model on device
        model = model.to(device)
        model.device = device

        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank])

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Define the loaders
        if isinstance(train_dataset, DataLoader):
            train_loader = train_dataset
            logger.warn(
                "Using the provided train dataloader! Carefull this may overwrite some "
                "parameters provided in your training config."
            )
        else:
            train_loader = self.get_train_dataloader(train_dataset)

        if eval_dataset is not None:
            if isinstance(eval_dataset, DataLoader):
                eval_loader = eval_dataset
                logger.warn(
                    "Using the provided eval dataloader! Carefull this may overwrite some "
                    "parameters provided in your training config."
                )
            else:
                eval_loader = self.get_eval_dataloader(eval_dataset)
        else:
            logger.info(
                "! No eval dataset provided ! -> keeping best model on train.\n"
            )
            self.training_config.keep_best_on_train = True
            eval_loader = None

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.callbacks = callbacks

        # run sanity check on the model
        self._run_model_sanity_check(model, train_loader)

        if self.is_main_process:
            logger.info("Model passed sanity check !\n" "Ready for training.\n")

        self.model = model

    @property
    def is_main_process(self):
        if self.rank == 0 or self.rank == -1:
            return True
        else:
            return False

    def _setup_devices(self):
        """Sets up the devices to perform distributed training."""

        if dist.is_available() and dist.is_initialized() and self.local_rank == -1:
            logger.warning(
                "torch.distributed process group is initialized, but local_rank == -1. "
            )
        if self.training_config.no_cuda:
            self._n_gpus = 0
            device = "cpu"

        else:
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)

            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.dist_backend,
                    init_method="env://",
                    world_size=self.world_size,
                    rank=self.rank,
                )

        return device

    def get_train_dataloader(
        self, train_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        if self.distributed:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=self.rank
            )
        else:
            train_sampler = None
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.training_config.per_device_train_batch_size,
            num_workers=self.training_config.train_dataloader_num_workers,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_dataset_output,
        )

    def get_eval_dataloader(
        self, eval_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        if self.distributed:
            eval_sampler = DistributedSampler(
                eval_dataset, num_replicas=self.world_size, rank=self.rank
            )
        else:
            eval_sampler = None
        return DataLoader(
            dataset=eval_dataset,
            batch_size=self.training_config.per_device_eval_batch_size,
            num_workers=self.training_config.eval_dataloader_num_workers,
            shuffle=(eval_sampler is None),
            sampler=eval_sampler,
            collate_fn=collate_dataset_output,
        )

    def set_optimizer(self):
        optimizer_cls = getattr(optim, self.training_config.optimizer_cls)

        if self.training_config.optimizer_params is not None:
            optimizer = optimizer_cls(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                **self.training_config.optimizer_params,
            )
        else:
            optimizer = optimizer_cls(
                self.model.parameters(), lr=self.training_config.learning_rate
            )

        self.optimizer = optimizer

    def set_scheduler(self):
        if self.training_config.scheduler_cls is not None:
            scheduler_cls = getattr(lr_scheduler, self.training_config.scheduler_cls)

            if self.training_config.scheduler_params is not None:
                scheduler = scheduler_cls(
                    self.optimizer, **self.training_config.scheduler_params
                )
            else:
                scheduler = scheduler_cls(self.optimizer)

        else:
            scheduler = None

        self.scheduler = scheduler

    def _set_output_dir(self):
        # Create folder
        if not os.path.exists(self.training_config.output_dir) and self.is_main_process:
            os.makedirs(self.training_config.output_dir, exist_ok=True)
            logger.info(
                f"Created {self.training_config.output_dir} folder since did not exist.\n"
            )

        self._training_signature = (
            str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        )

        training_dir = os.path.join(
            self.training_config.output_dir,
            f"{self.model_name}_training_{self._training_signature}",
        )

        self.training_dir = training_dir

        if not os.path.exists(training_dir) and self.is_main_process:
            os.makedirs(training_dir, exist_ok=True)
            logger.info(
                f"Created {training_dir}. \n"
                "Training config, checkpoints and final model will be saved here.\n"
            )

    def _get_file_logger(self, log_output_dir):
        log_dir = log_output_dir

        # if dir does not exist create it
        if not os.path.exists(log_dir) and self.is_main_process:
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"Created {log_dir} folder since did not exists.")
            logger.info("Training logs will be recodered here.\n")
            logger.info(" -> Training can be monitored here.\n")

        # create and set logger
        log_name = f"training_logs_{self._training_signature}"

        file_logger = logging.getLogger(log_name)
        file_logger.setLevel(logging.INFO)
        f_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_logs_{self._training_signature}.log")
        )
        f_handler.setLevel(logging.INFO)
        file_logger.addHandler(f_handler)

        # Do not output logs in the console
        file_logger.propagate = False

        return file_logger

    def _setup_callbacks(self):
        if self.callbacks is None:
            self.callbacks = [TrainingCallback()]

        self.callback_handler = CallbackHandler(
            callbacks=self.callbacks, model=self.model
        )

        self.callback_handler.add_callback(ProgressBarCallback())
        self.callback_handler.add_callback(MetricConsolePrinterCallback())

    def _run_model_sanity_check(self, model, loader):
        try:
            inputs = next(iter(loader))
            train_dataset = self._set_inputs_to_device(inputs)
            model(train_dataset)

        except Exception as e:
            raise ModelError(
                "Error when calling forward method from model. Potential issues: \n"
                " - Wrong model architecture -> check encoder, decoder and metric architecture if "
                "you provide yours \n"
                " - The data input dimension provided is wrong -> when no encoder, decoder or metric "
                "provided, a network is built automatically but requires the shape of the flatten "
                "input data.\n"
                f"Exception raised: {type(e)} with message: " + str(e)
            ) from e

    def _set_optimizer_on_device(self, optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

        return optim

    def _set_inputs_to_device(self, inputs: Dict[str, Any]):

        inputs_on_device = inputs

        if self.device == "cuda":
            cuda_inputs = dict.fromkeys(inputs)

            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    cuda_inputs[key] = inputs[key].cuda()

                else:
                    cuda_inputs[key] = inputs[key]
            inputs_on_device = cuda_inputs

        return inputs_on_device

    def _optimizers_step(self, model_output=None):

        loss = model_output.loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _schedulers_step(self, metrics=None):
        if self.scheduler is None:
            pass

        elif isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics)

        else:
            self.scheduler.step()

    def prepare_training(self):
        """Sets up the trainer for training"""
        # set random seed
        set_seed(self.training_config.seed)

        # set optimizer
        self.set_optimizer()

        # set scheduler
        self.set_scheduler()

        # create folder for saving
        self._set_output_dir()

        # set callbacks
        self._setup_callbacks()

    def train(self, log_output_dir: str = None):
        """This function is the main training function

        Args:
            log_output_dir (str): The path in which the log will be stored
        """

        self.prepare_training()

        self.callback_handler.on_train_begin(
            training_config=self.training_config, model_config=self.model_config
        )

        log_verbose = False

        msg = (
            f"Training params:\n - max_epochs: {self.training_config.num_epochs}\n"
            " - per_device_train_batch_size: "
            f"{self.training_config.per_device_train_batch_size}\n"
            " - per_device_eval_batch_size: "
            f"{self.training_config.per_device_eval_batch_size}\n"
            f" - checkpoint saving every: {self.training_config.steps_saving}\n"
            f"Optimizer: {self.optimizer}\n"
            f"Scheduler: {self.scheduler}\n"
        )

        if self.is_main_process:
            logger.info(msg)

        # set up log file
        if log_output_dir is not None and self.is_main_process:
            log_verbose = True
            file_logger = self._get_file_logger(log_output_dir=log_output_dir)

            file_logger.info(msg)

        if self.is_main_process:
            logger.info("Successfully launched training !\n")

        # set best losses for early stopping
        best_train_loss = 1e10
        best_eval_loss = 1e10

        for epoch in range(1, self.training_config.num_epochs + 1):

            self.callback_handler.on_epoch_begin(
                training_config=self.training_config,
                epoch=epoch,
                train_loader=self.train_loader,
                eval_loader=self.eval_loader,
            )

            metrics = {}

            epoch_train_loss = self.train_step(epoch)
            metrics["train_epoch_loss"] = epoch_train_loss

            if self.eval_dataset is not None:
                epoch_eval_loss = self.eval_step(epoch)
                metrics["eval_epoch_loss"] = epoch_eval_loss
                self._schedulers_step(epoch_eval_loss)

            else:
                epoch_eval_loss = best_eval_loss
                self._schedulers_step(epoch_train_loss)

            if (
                epoch_eval_loss < best_eval_loss
                and not self.training_config.keep_best_on_train
            ):
                best_eval_loss = epoch_eval_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            elif (
                epoch_train_loss < best_train_loss
                and self.training_config.keep_best_on_train
            ):
                best_train_loss = epoch_train_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            if (
                self.training_config.steps_predict is not None
                and epoch % self.training_config.steps_predict == 0
                and self.is_main_process
            ):
                true_data, reconstructions, generations = self.predict(best_model)

                self.callback_handler.on_prediction_step(
                    self.training_config,
                    true_data=true_data,
                    reconstructions=reconstructions,
                    generations=generations,
                    global_step=epoch,
                )

            self.callback_handler.on_epoch_end(training_config=self.training_config)

            # save checkpoints
            if (
                self.training_config.steps_saving is not None
                and epoch % self.training_config.steps_saving == 0
            ):
                if self.is_main_process:
                    self.save_checkpoint(
                        model=best_model, dir_path=self.training_dir, epoch=epoch
                    )
                    logger.info(f"Saved checkpoint at epoch {epoch}\n")

                    if log_verbose:
                        file_logger.info(f"Saved checkpoint at epoch {epoch}\n")

            self.callback_handler.on_log(
                self.training_config,
                metrics,
                logger=logger,
                global_step=epoch,
                rank=self.rank,
            )

        final_dir = os.path.join(self.training_dir, "final_model")

        if self.is_main_process:
            self.save_model(best_model, dir_path=final_dir)

            logger.info("Training ended!")
            logger.info(f"Saved final model in {final_dir}")

        if self.distributed:
            dist.destroy_process_group()

        self.callback_handler.on_train_end(self.training_config)

    def eval_step(self, epoch: int):
        """Perform an evaluation step

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The evaluation loss
        """

        self.callback_handler.on_eval_step_begin(
            training_config=self.training_config,
            eval_loader=self.eval_loader,
            epoch=epoch,
            rank=self.rank,
        )

        self.model.eval()

        epoch_loss = 0

        with self.amp_context:
            for inputs in self.eval_loader:

                inputs = self._set_inputs_to_device(inputs)

                try:
                    with torch.no_grad():

                        model_output = self.model(
                            inputs,
                            epoch=epoch,
                            dataset_size=len(self.eval_loader.dataset),
                            uses_ddp=self.distributed,
                        )

                except RuntimeError:
                    model_output = self.model(
                        inputs,
                        epoch=epoch,
                        dataset_size=len(self.eval_loader.dataset),
                        uses_ddp=self.distributed,
                    )

                loss = model_output.loss

                epoch_loss += loss.item()

                if epoch_loss != epoch_loss:
                    raise ArithmeticError("NaN detected in eval loss")

                self.callback_handler.on_eval_step_end(
                    training_config=self.training_config
                )

        epoch_loss /= len(self.eval_loader)

        return epoch_loss

    def train_step(self, epoch: int):
        """The trainer performs training loop over the train_loader.

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The step training loss
        """
        self.callback_handler.on_train_step_begin(
            training_config=self.training_config,
            train_loader=self.train_loader,
            epoch=epoch,
            rank=self.rank,
        )

        # set model in train model
        self.model.train()

        epoch_loss = 0

        for inputs in self.train_loader:

            inputs = self._set_inputs_to_device(inputs)

            with self.amp_context:
                model_output = self.model(
                    inputs,
                    epoch=epoch,
                    dataset_size=len(self.train_loader.dataset),
                    uses_ddp=self.distributed,
                )

            self._optimizers_step(model_output)

            loss = model_output.loss

            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(
                training_config=self.training_config
            )

        # Allows model updates if needed
        if self.distributed:
            self.model.module.update()
        else:
            self.model.update()

        epoch_loss /= len(self.train_loader)

        return epoch_loss

    def save_model(self, model: BaseAE, dir_path: str):
        """This method saves the final model along with the config files

        Args:
            model (BaseAE): The model to be saved
            dir_path (str): The folder where the model and config files should be saved
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save model
        if self.distributed:
            model.module.save(dir_path)

        else:
            model.save(dir_path)

        # save training config
        self.training_config.save_json(dir_path, "training_config")

        self.callback_handler.on_save(self.training_config)

    def save_checkpoint(self, model: BaseAE, dir_path, epoch: int):
        """Saves a checkpoint alowing to restart training from here

        Args:
            dir_path (str): The folder where the checkpoint should be saved
            epochs_signature (int): The epoch number"""

        checkpoint_dir = os.path.join(dir_path, f"checkpoint_epoch_{epoch}")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # save optimizer
        torch.save(
            deepcopy(self.optimizer.state_dict()),
            os.path.join(checkpoint_dir, "optimizer.pt"),
        )

        # save scheduler
        if self.scheduler is not None:
            torch.save(
                deepcopy(self.scheduler.state_dict()),
                os.path.join(checkpoint_dir, "scheduler.pt"),
            )

        # save model
        if self.distributed:
            model.module.save(checkpoint_dir)

        else:
            model.save(checkpoint_dir)

        # save training config
        self.training_config.save_json(checkpoint_dir, "training_config")

    def predict(self, model: BaseAE):

        model.eval()

        with self.amp_context:
            inputs = next(iter(self.eval_loader))
            inputs = self._set_inputs_to_device(inputs)

            model_out = model(inputs)
            reconstructions = model_out.recon_x.cpu().detach()[
                : min(inputs["data"].shape[0], 10)
            ]
            z_enc = model_out.z[: min(inputs["data"].shape[0], 10)]
            z = torch.randn_like(z_enc)
            if self.distributed:
                normal_generation = (
                    model.module.decoder(z).reconstruction.detach().cpu()
                )
            else:
                normal_generation = model.decoder(z).reconstruction.detach().cpu()

        return (
            inputs["data"][: min(inputs["data"].shape[0], 10)],
            reconstructions,
            normal_generation,
        )
