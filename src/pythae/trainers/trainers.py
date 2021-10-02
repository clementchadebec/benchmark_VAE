import datetime
import logging
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from pyraug.customexception import ModelError
from pyraug.data.datasets import BaseDataset
from pyraug.models import BaseVAE
from pyraug.trainers.trainer_utils import set_seed
from pyraug.trainers.training_config import TrainingConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class Trainer:
    """Trainer is the main class to perform model training.

    Args:
        model (BaseVAE): The model to train

        train_dataset (BaseDataset): The training dataset of type :class:`~pyraug.`

        training_args (TrainingConfig): The training arguments summarizing the main parameters used
            for training. If None, a basic training instance of :class:`TrainingConfig` is used.
            Default: None.

        optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer` used for
            training. If None, a :class:`~torch.optim.Adam` optimizer is used. Default: None.
    """

    def __init__(
        self,
        model: BaseVAE,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
        training_config: Optional[TrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):

        if training_config is None:
            training_config = TrainingConfig()

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

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.model = model
        self.optimizer = optimizer

        self.device = device

        # set early stopping flags
        self._set_earlystopping_flags(train_dataset, eval_dataset, training_config)

        # Define the loaders
        train_loader = self.get_train_dataloader(train_dataset)

        if eval_dataset is not None:
            eval_loader = self.get_eval_dataloader(eval_dataset)

        else:
            eval_loader = None

        self.train_loader = train_loader
        self.eval_loader = eval_loader

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

    def set_default_optimizer(self, model: BaseVAE) -> torch.optim.Optimizer:

        optimizer = optim.Adam(
            model.parameters(), lr=self.training_config.learning_rate
        )

        return optimizer

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

    #
    def _set_earlystopping_flags(self, train_dataset, eval_dataset, training_config):

        # Initialize early_stopping flags
        self.make_eval_early_stopping = False
        self.make_train_early_stopping = False

        if training_config.train_early_stopping is not None:
            self.make_train_early_stopping = True

        # Check if eval_dataset is provided
        if eval_dataset is not None and training_config.eval_early_stopping is not None:
            self.make_eval_early_stopping = True

            # By default we make the early stopping on evaluation dataset
            self.make_train_early_stopping = False

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
            self.training_config.output_dir, f"training_{self._training_signature}"
        )

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
                f"Training params:\n - max_epochs: {self.training_config.max_epochs}\n"
                f" - train es: {self.training_config.train_early_stopping}\n"
                f" - eval es: {self.training_config.eval_early_stopping}\n"
                f" - batch_size: {self.training_config.batch_size}\n"
                f" - checkpoint saving every {self.training_config.steps_saving}\n"
            )

            file_logger.info(f"Model Architecture: {self.model}\n")
            file_logger.info(f"Optimizer: {self.optimizer}\n")

        logger.info("Successfully launched training !")

        # set best losses for early stopping
        best_train_loss = 1e10
        best_eval_loss = 1e10

        epoch_es_train = 0
        epoch_es_eval = 0

        for epoch in range(1, self.training_config.max_epochs):

            epoch_train_loss = self.train_step()

            if self.eval_dataset is not None:
                epoch_eval_loss = self.eval_step()

            # early stopping
            if self.make_eval_early_stopping:

                if epoch_eval_loss < best_eval_loss:
                    epoch_es_eval = 0
                    best_eval_loss = epoch_eval_loss

                else:
                    epoch_es_eval += 1

                    if (
                        epoch_es_eval >= self.training_config.eval_early_stopping
                        and log_verbose
                    ):
                        logger.info(
                            f"Training ended at epoch {epoch}! "
                            f" Eval loss did not improve for {epoch_es_eval} epochs."
                        )
                        file_logger.info(
                            f"Training ended at epoch {epoch}! "
                            f" Eval loss did not improve for {epoch_es_eval} epochs."
                        )

                        break

            elif self.make_train_early_stopping:

                if epoch_train_loss < best_train_loss:
                    epoch_es_train = 0
                    best_train_loss = epoch_train_loss

                else:
                    epoch_es_train += 1

                    if (
                        epoch_es_train >= self.training_config.train_early_stopping
                        and log_verbose
                    ):
                        logger.info(
                            f"Training ended at epoch {epoch}! "
                            f" Train loss did not improve for {epoch_es_train} epochs."
                        )
                        file_logger.info(
                            f"Training ended at epoch {epoch}! "
                            f" Train loss did not improve for {epoch_es_train} epochs."
                        )

                        break

            # save checkpoints
            if (
                self.training_config.steps_saving is not None
                and epoch % self.training_config.steps_saving == 0
            ):
                self.save_checkpoint(dir_path=training_dir, epoch=epoch)
                logger.info(f"Saved checkpoint at epoch {epoch}\n")

                if log_verbose:
                    file_logger.info(f"Saved checkpoint at epoch {epoch}\n")

            if log_verbose and epoch % 10 == 0:
                if self.eval_dataset is not None:
                    if self.make_eval_early_stopping:
                        file_logger.info(
                            f"Epoch {epoch} / {self.training_config.max_epochs}\n"
                            f"- Current Train loss: {epoch_train_loss:.2f}\n"
                            f"- Current Eval loss: {epoch_eval_loss:.2f}\n"
                            f"- Eval Early Stopping: {epoch_es_eval}/{self.training_config.eval_early_stopping}"
                            f" (Best: {best_eval_loss:.2f})\n"
                        )
                    elif self.make_train_early_stopping:
                        file_logger.info(
                            f"Epoch {epoch} / {self.training_config.max_epochs}\n"
                            f"- Current Train loss: {epoch_train_loss:.2f}\n"
                            f"- Current Eval loss: {epoch_eval_loss:.2f}\n"
                            f"- Train Early Stopping: {epoch_es_train}/{self.training_config.train_early_stopping}"
                            f" (Best: {best_train_loss:.2f})\n"
                        )

                    else:
                        file_logger.info(
                            f"Epoch {epoch} / {self.training_config.max_epochs}\n"
                            f"- Current Train loss: {epoch_train_loss:.2f}\n"
                            f"- Current Eval loss: {epoch_eval_loss:.2f}\n"
                        )
                else:
                    if self.make_train_early_stopping:
                        file_logger.info(
                            f"Epoch {epoch} / {self.training_config.max_epochs}\n"
                            f"- Current Train loss: {epoch_train_loss:.2f}\n"
                            f"- Train Early Stopping: {epoch_es_train}/{self.training_config.train_early_stopping}"
                            f" (Best: {best_train_loss:.2f})\n"
                        )

                    else:
                        file_logger.info(
                            f"Epoch {epoch} / {self.training_config.max_epochs}\n"
                            f"- Current Train loss: {epoch_train_loss:.2f}\n"
                        )

        final_dir = os.path.join(training_dir, "final_model")

        self.save_model(dir_path=final_dir)
        logger.info("----------------------------------")
        logger.info("Training ended!")
        logger.info(f"Saved final model in {final_dir}")

    def eval_step(self):
        """Perform an evaluation step

        Returns:
            (torch.Tensor): The evaluation loss
        """

        self.model.eval()

        epoch_loss = 0

        for (batch_idx, inputs) in enumerate(self.eval_loader):

            inputs = self._set_inputs_to_device(inputs)

            model_output = self.model(inputs)

            loss = model_output.loss

            epoch_loss += loss.item()

        epoch_loss /= len(self.eval_loader)

        return epoch_loss

    def train_step(self):
        """The trainer performs training loop over the train_loader.

        Returns:
            (torch.Tensor): The step training loss
        """
        # set model in train model
        self.model.train()

        epoch_loss = 0

        for (batch_idx, inputs) in enumerate(self.train_loader):

            inputs = self._set_inputs_to_device(inputs)

            self.optimizer.zero_grad()

            model_output = self.model(inputs)

            loss = model_output.loss

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        # Allows model updates if needed
        self.model.update()

        epoch_loss /= len(self.train_loader)

        return epoch_loss

    def save_model(self, dir_path):
        """This method saves the final model along with the config files

        Args:
            dir_path (str): The folder where the model and config files should be saved
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save model
        self.model.save(dir_path)

        # save training config
        self.training_config.save_json(dir_path, "training_config")

    def save_checkpoint(self, dir_path, epoch: int):
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

        # save model
        self.model.save(checkpoint_dir)

        # save training config
        self.training_config.save_json(checkpoint_dir, "training_config")
