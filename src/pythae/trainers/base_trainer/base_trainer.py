import datetime
import logging
import os
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Optional
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ...customexception import ModelError
from ...data.datasets import BaseDataset
from ...models import BaseAE
from ..trainer_utils import set_seed
from .base_training_config import BaseTrainingConfig
from ..training_callbacks import (
    TrainingCallback,
    CallbackHandler,
    WandbCallback
)

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseTrainer:
    """Base class to perform model training.

    Args:
        model (BaseAE): The model to train

        train_dataset (BaseDataset): The training dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        training_args (BaseTrainingConfig): The training arguments summarizing the main parameters used
            for training. If None, a basic training instance of :class:`TrainingConfig` is used.
            Default: None.

        optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer` used for
            training. If None, a :class:`~torch.optim.Adam` optimizer is used. Default: None.
    """

    def __init__(
        self,
        model: BaseAE,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
        training_config: Optional[BaseTrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional = None,
        callbacks: List[TrainingCallback] = None
    ):

        if training_config is None:
            training_config = BaseTrainingConfig()

        if training_config.output_dir is None:
            output_dir = "dummy_output_dir"
            training_config.output_dir = output_dir

        if not os.path.exists(training_config.output_dir):
            os.makedirs(training_config.output_dir)
            logger.info(
                f"Created {training_config.output_dir} folder since did not exist.\n"
            )

        self.training_config = training_config

        set_seed(self.training_config.seed)

        device = (
            "cuda"
            if torch.cuda.is_available() and not training_config.no_cuda
            else "cpu"
        )

        # place model on device
        model = model.to(device)
        model.device = device

        # set optimizer
        if optimizer is None:
            optimizer = self.set_default_optimizer(model)

        else:
            optimizer = self._set_optimizer_on_device(optimizer, device)

        # set scheduler
        if scheduler is None:
            scheduler = self.set_default_scheduler(model, optimizer)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device

        # Define the loaders
        train_loader = self.get_train_dataloader(train_dataset)

        if eval_dataset is not None:
            eval_loader = self.get_eval_dataloader(eval_dataset)

        else:
            logger.info(
                "! No eval dataset provided ! -> keeping best model on train.\n"
            )
            self.training_config.keep_best_on_train = True
            eval_loader = None

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        if callbacks is None:
            callbacks = TrainingCallback()

        self.callback_handler = CallbackHandler(
            callbacks=callbacks, model=model, optimier=optimizer, scheduler=scheduler)

    def get_train_dataloader(
        self, train_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:

        return DataLoader(
            dataset=train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )

    def get_eval_dataloader(
        self, eval_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=eval_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
        )

    def set_default_optimizer(self, model: BaseAE) -> torch.optim.Optimizer:

        optimizer = optim.Adam(
            model.parameters(), lr=self.training_config.learning_rate
        )

        return optimizer

    def set_default_scheduler(
        self, model: BaseAE, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler:

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10, verbose=True
        )

        return scheduler

    def _run_model_sanity_check(self, model, train_dataset):
        try:
            train_dataset = self._set_inputs_to_device(train_dataset[:2])
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
                    cuda_inputs = inputs[key]
            inputs_on_device = cuda_inputs

        return inputs_on_device

    def train(self, log_output_dir: str = None):
        """This function is the main training function

        Args:
            log_output_dir (str): The path in which the log will be stored
        """

        # run sanity check on the model
        self._run_model_sanity_check(self.model, self.train_dataset)

        logger.info("Model passed sanity check !\n")

        self._training_signature = (
            str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        )

        training_dir = os.path.join(
            self.training_config.output_dir,
            f"{self.model.model_name}_training_{self._training_signature}",
        )

        self.training_dir = training_dir

        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
            logger.info(
                f"Created {training_dir}. \n"
                "Training config, checkpoints and final model will be saved here.\n"
            )

        log_verbose = False

        # set up log file
        if log_output_dir is not None:
            log_dir = log_output_dir
            log_verbose = True

            # if dir does not exist create it
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
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

            file_logger.info("Training started !\n")
            file_logger.info(
                f"Training params:\n - max_epochs: {self.training_config.num_epochs}\n"
                f" - batch_size: {self.training_config.batch_size}\n"
                f" - checkpoint saving every {self.training_config.steps_saving}\n"
            )

            file_logger.info(f"Model Architecture: {self.model}\n")
            file_logger.info(f"Optimizer: {self.optimizer}\n")

        logger.info("Successfully launched training !\n")

        # set best losses for early stopping
        best_train_loss = 1e10
        best_eval_loss = 1e10

        for epoch in range(1, self.training_config.num_epochs+1):

            epoch_train_loss = self.train_step(epoch)

            if self.eval_dataset is not None:
                epoch_eval_loss = self.eval_step(epoch)
                self.scheduler.step(epoch_eval_loss)

            else:
                epoch_eval_loss = best_eval_loss
                self.scheduler.step(epoch_train_loss)

            if (
                epoch_eval_loss < best_eval_loss
                and not self.training_config.keep_best_on_train
            ):
                best_model_epoch = epoch
                best_eval_loss = epoch_eval_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            elif (
                epoch_train_loss < best_train_loss
                and self.training_config.keep_best_on_train
            ):
                best_model_epoch = epoch
                best_train_loss = epoch_train_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model

            # save checkpoints
            if (
                self.training_config.steps_saving is not None
                and epoch % self.training_config.steps_saving == 0
            ):
                self.save_checkpoint(model=best_model, dir_path=training_dir, epoch=epoch)
                logger.info(f"Saved checkpoint at epoch {epoch}\n")

                if log_verbose:
                    file_logger.info(f"Saved checkpoint at epoch {epoch}\n")

            if self.eval_dataset is not None:
                logger.info(
                    "----------------------------------------------------------------"
                )
                logger.info(
                    f"Epoch {epoch}: Train loss: {np.round(epoch_train_loss, 10)}"
                )
                logger.info(
                    f"Epoch {epoch}: Eval loss: {np.round(epoch_eval_loss, 10)}"
                )
                logger.info(
                    "----------------------------------------------------------------"
                )

            else:
                logger.info(
                    "----------------------------------------------------------------"
                )
                logger.info(
                    f"Epoch {epoch}: Train loss: {np.round(epoch_train_loss, 10)}"
                )
                logger.info(
                    "----------------------------------------------------------------"
                )

        final_dir = os.path.join(training_dir, "final_model")

        self.save_model(best_model, dir_path=final_dir)
        logger.info("----------------------------------")
        logger.info("Training ended!")
        logger.info(f"Saved final model in {final_dir}")

    def eval_step(self, epoch: int):
        """Perform an evaluation step

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The evaluation loss
        """

        self.model.eval()

        epoch_loss = 0

        with tqdm(self.eval_loader, unit="batch") as tepoch:
            for inputs in tepoch:

                tepoch.set_description(
                    f"Eval of epoch {epoch}/{self.training_config.num_epochs}"
                )

                #update epoch in model
                self.model.epoch = epoch

                inputs = self._set_inputs_to_device(inputs)

                model_output = self.model(
                    inputs, epoch=epoch, dataset_size=len(self.eval_loader.dataset)
                )

                loss = model_output.loss

                epoch_loss += loss.item()

                if epoch_loss != epoch_loss:
                    raise ArithmeticError("NaN detected in eval loss")

            epoch_loss /= len(self.eval_loader)

        return epoch_loss

    def train_step(self, epoch: int):
        """The trainer performs training loop over the train_loader.

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The step training loss
        """
        # set model in train model
        self.model.train()

        epoch_loss = 0

        with tqdm(self.train_loader, unit="batch") as tepoch:
            for inputs in tepoch:

                tepoch.set_description(
                    f"Training of epoch {epoch}/{self.training_config.num_epochs}"
                )

                inputs = self._set_inputs_to_device(inputs)

                self.optimizer.zero_grad()

                model_output = self.model(
                    inputs, epoch=epoch, dataset_size=len(self.train_loader.dataset)
                )

                loss = model_output.loss

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if epoch_loss != epoch_loss:
                    raise ArithmeticError("NaN detected in train loss")

            # Allows model updates if needed
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
        model.save(dir_path)

        # save training config
        self.training_config.save_json(dir_path, "training_config")

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
        torch.save(
            deepcopy(self.scheduler.state_dict()),
            os.path.join(checkpoint_dir, "scheduler.pt"),
        )

        # save model
        model.save(checkpoint_dir)

        # save training config
        self.training_config.save_json(checkpoint_dir, "training_config")
