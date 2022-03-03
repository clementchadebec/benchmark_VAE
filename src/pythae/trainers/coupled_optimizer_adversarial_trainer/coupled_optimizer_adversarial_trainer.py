import datetime
import logging
import os
from copy import deepcopy
from typing import Any, Dict, Optional
from tqdm import tqdm
import itertools
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ...customexception import ModelError
from ...data.datasets import BaseDataset
from ...models import BaseAE
from ..trainer_utils import set_seed
from ...trainers import AdversarialTrainer, CoupledOptimizerTrainer, BaseTrainer
from .coupled_optimizer_adversarial_trainer_config import CoupledOptimizerAdversarialTrainerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class CoupledOptimizerAdversarialTrainer(BaseTrainer):
    """Trainer using disctinct optimizers for the encoder, decoder and discriminator.

    Args:
        model (BaseAE): The model to train

        train_dataset (BaseDataset): The training dataset of type 
            :class:`~pythae.data.dataset.BaseDataset`

        training_args (CoupledOptimizerAdversarialTrainerConfig): The training arguments summarizing
            the main parameters used for training. If None, a basic training instance of 
            :class:`AdversarialTrainerConfig` is used. Default: None.

        encoder_optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer` 
            used for training the encoder. If None, a :class:`~torch.optim.Adam` optimizer is 
            used. Default: None.

        decoder_optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer` 
            used for training the decoder. If None, a :class:`~torch.optim.Adam` optimizer is 
            used. Default: None.

        discriminator_optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer` 
            used for training the discriminator. If None, a :class:`~torch.optim.Adam` optimizer is 
            used. Default: None.
    """

    def __init__(
        self,
        model: BaseAE,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
        training_config: Optional[CoupledOptimizerAdversarialTrainerConfig] = None,
        encoder_optimizer: Optional[torch.optim.Optimizer] = None,
        decoder_optimizer: Optional[torch.optim.Optimizer] = None,
        discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
        encoder_scheduler: Optional = None,
        decoder_scheduler: Optional = None,
        discriminator_scheduler: Optional = None
    ):

        BaseTrainer.__init__(
            self,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
            optimizer=None)


        # set encoder optimizer
        if encoder_optimizer is None:
            encoder_optimizer = self.set_default_encoder_optimizer(model)

        else:
            encoder_optimizer = self._set_optimizer_on_device(
                encoder_optimizer, self.device
            )

        if encoder_scheduler is None:
            encoder_scheduler = self.set_default_scheduler(model, encoder_optimizer)

        # set decoder optimizer
        if decoder_optimizer is None:
            decoder_optimizer = self.set_default_decoder_optimizer(model)

        else:
            decoder_optimizer = self._set_optimizer_on_device(
                decoder_optimizer, self.device
            )

        if decoder_scheduler is None:
            decoder_scheduler = self.set_default_scheduler(model, decoder_optimizer)

        # set decoder optimizer
        if discriminator_optimizer is None:
            discriminator_optimizer = self.set_default_discriminator_optimizer(model)

        else:
            discriminator_optimizer = self._set_optimizer_on_device(
                discriminator_optimizer, self.device
            )

        if discriminator_scheduler is None:
            discriminator_scheduler = self.set_default_scheduler(model, discriminator_optimizer)

        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.encoder_scheduler = encoder_scheduler
        self.decoder_scheduler = decoder_scheduler
        self.discriminator_scheduler = discriminator_scheduler

        self.optimizer = None

    def set_default_encoder_optimizer(self, model: BaseAE) -> torch.optim.Optimizer:

        optimizer = optim.Adam(
            model.encoder.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.encoder_optim_decay
        )

        return optimizer

    def set_default_decoder_optimizer(self, model: BaseAE) -> torch.optim.Optimizer:

        optimizer = optim.Adam(
            model.decoder.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.decoder_optim_decay
        )

        return optimizer

    def set_default_discriminator_optimizer(self, model: BaseAE) -> torch.optim.Optimizer:

        optimizer = optim.Adam(
            model.discriminator.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.discriminator_optim_decay
        )

        return optimizer


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

            train_losses = self.train_step(epoch)

            [
                epoch_train_loss,
                epoch_train_encoder_loss,
                epoch_train_decoder_loss,
                epoch_train_discriminator_loss
            ] = train_losses

            if self.eval_dataset is not None:
                eval_losses = self.eval_step(epoch)

                [
                    epoch_eval_loss,
                    epoch_eval_encoder_loss,
                    epoch_eval_decoder_loss,
                    epoch_eval_discriminator_loss
                ] = eval_losses

                self.encoder_scheduler.step(epoch_eval_encoder_loss)
                self.decoder_scheduler.step(epoch_eval_decoder_loss)
                self.discriminator_scheduler.step(epoch_eval_discriminator_loss)

            else:
                epoch_eval_loss = best_eval_loss
                self.encoder_scheduler.step(epoch_train_encoder_loss)
                self.decoder_scheduler.step(epoch_train_decoder_loss)
                self.discriminator_scheduler.step(epoch_train_discriminator_loss)

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
                    f"Epoch {epoch}: Train loss: {np.round(epoch_train_loss, 10)}\t "
                    f"Encoder loss: {np.round(epoch_train_encoder_loss, 10)}\t "
                    f"Decoder loss: {np.round(epoch_train_decoder_loss, 10)}\t "
                    f"Discriminator loss: {np.round(epoch_train_discriminator_loss, 10)}"
                )
                logger.info(
                    f"Epoch {epoch}: Eval loss: {np.round(epoch_eval_loss, 10)}\t "
                    f"Encoder loss: {np.round(epoch_eval_encoder_loss, 10)}\t "
                    f"Decoder loss: {np.round(epoch_eval_decoder_loss, 10)}\t "
                    f"Discriminator loss: {np.round(epoch_eval_discriminator_loss, 10)}"
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

        epoch_encoder_loss = 0
        epoch_decoder_loss = 0
        epoch_discriminator_loss = 0
        epoch_loss = 0

        with tqdm(self.eval_loader, unit="batch") as tepoch:
            for i, inputs in enumerate(tepoch):

                tepoch.set_description(
                    f"Eval of epoch {epoch}/{self.training_config.num_epochs}"
                )

                inputs = self._set_inputs_to_device(inputs)

                model_output = self.model(
                    inputs, epoch=epoch, dataset_size=len(self.eval_loader.dataset)
                )

                encoder_loss = model_output.encoder_loss
                decoder_loss = model_output.decoder_loss
                discriminator_loss = model_output.discriminator_loss

                loss = encoder_loss + decoder_loss + discriminator_loss

                epoch_encoder_loss += encoder_loss.item()
                epoch_decoder_loss += decoder_loss.item()
                epoch_discriminator_loss += discriminator_loss.item()
                epoch_loss += loss.item()

                if epoch_loss != epoch_loss:
                    raise ArithmeticError("NaN detected in eval loss")

                tepoch.set_postfix(
                    {
                        'encoder_loss': epoch_encoder_loss / (i+1),
                        'decoder_loss': epoch_decoder_loss / (i+1),
                        'discriminator_loss': epoch_discriminator_loss / (i+1)
                    }
                )

            epoch_encoder_loss /= len(self.eval_loader)
            epoch_decoder_loss /= len(self.eval_loader)
            epoch_discriminator_loss /= len(self.eval_loader) 
            epoch_loss /= len(self.eval_loader)

        return epoch_loss, epoch_encoder_loss, epoch_decoder_loss, epoch_discriminator_loss

    def train_step(self, epoch: int):
        """The trainer performs training loop over the train_loader.

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The step training loss
        """
        # set model in train model
        self.model.train()

        epoch_encoder_loss = 0
        epoch_decoder_loss = 0
        epoch_discriminator_loss = 0
        epoch_loss = 0

        with tqdm(self.train_loader, unit="batch") as tepoch:
            for i, inputs in enumerate(tepoch):

                tepoch.set_description(
                    f"Training of epoch {epoch}/{self.training_config.num_epochs}"
                )

                inputs = self._set_inputs_to_device(inputs)

                model_output = self.model(
                    inputs, epoch=epoch, dataset_size=len(self.train_loader.dataset)
                )

                encoder_loss = model_output.encoder_loss
                decoder_loss = model_output.decoder_loss
                discriminator_loss = model_output.discriminator_loss
                
                # Reset optimizers
                if model_output.update_encoder:
                    self.encoder_optimizer.zero_grad()
                    encoder_loss.backward(retain_graph=True)
               
                if model_output.update_decoder:
                    self.decoder_optimizer.zero_grad()
                    decoder_loss.backward(retain_graph=True)
                
                if model_output.update_discriminator:
                    self.discriminator_optimizer.zero_grad()
                    discriminator_loss.backward()

                self.encoder_optimizer.step()

                if model_output.update_decoder:
                    #print('update dec')
                    self.decoder_optimizer.step()

                if model_output.update_discriminator:
                    #print('update discr')
                    self.discriminator_optimizer.step()

                loss = encoder_loss + decoder_loss + discriminator_loss

                epoch_encoder_loss += encoder_loss.item()
                epoch_decoder_loss += decoder_loss.item()
                epoch_discriminator_loss += discriminator_loss.item()
                epoch_loss += loss.item()

                tepoch.set_postfix(
                    {
                        'encoder_loss': epoch_encoder_loss / (i+1),
                        'decoder_loss': epoch_decoder_loss / (i+1),
                        'discriminator_loss': epoch_discriminator_loss / (i+1)
                    }
                )

            # Allows model updates if needed
            self.model.update()

            epoch_encoder_loss /= len(self.train_loader)
            epoch_decoder_loss /= len(self.train_loader)
            epoch_discriminator_loss /= len(self.train_loader) 
            epoch_loss /= len(self.train_loader)

        return epoch_loss, epoch_encoder_loss, epoch_decoder_loss, epoch_discriminator_loss


    def save_checkpoint(self, model: BaseAE, dir_path, epoch: int):
        """Saves a checkpoint alowing to restart training from here

        Args:
            dir_path (str): The folder where the checkpoint should be saved
            epochs_signature (int): The epoch number"""

        checkpoint_dir = os.path.join(dir_path, f"checkpoint_epoch_{epoch}")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # save optimizers
        torch.save(
            deepcopy(self.encoder_optimizer.state_dict()),
            os.path.join(checkpoint_dir, "encoder_optimizer.pt"),
        )
        torch.save(
            deepcopy(self.decoder_optimizer.state_dict()),
            os.path.join(checkpoint_dir, "decoder_optimizer.pt"),
        )
        torch.save(
            deepcopy(self.discriminator_optimizer.state_dict()),
            os.path.join(checkpoint_dir, "discriminator_optimizer.pt"),
        )

        # save model
        model.save(checkpoint_dir)

        # save training config
        self.training_config.save_json(checkpoint_dir, "training_config")
