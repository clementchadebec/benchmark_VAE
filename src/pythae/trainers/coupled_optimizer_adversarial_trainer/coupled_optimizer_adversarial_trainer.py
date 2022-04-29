import datetime
import dis
import logging
import os
from copy import deepcopy
from typing import List, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ...data.datasets import BaseDataset
from ...models import BaseAE
from ..base_trainer import BaseTrainer
from ..training_callbacks import TrainingCallback
from .coupled_optimizer_adversarial_trainer_config import (
    CoupledOptimizerAdversarialTrainerConfig,
)

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class CoupledOptimizerAdversarialTrainer(BaseTrainer):
    """Trainer using distinct optimizers for the encoder, decoder and discriminator.

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
        encoder_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        decoder_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        discriminator_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        callbacks: List[TrainingCallback] = None,
    ):

        BaseTrainer.__init__(
            self,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
            optimizer=None,
            callbacks=callbacks,
        )

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
            discriminator_scheduler = self.set_default_scheduler(
                model, discriminator_optimizer
            )

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
            weight_decay=self.training_config.encoder_optim_decay,
        )

        return optimizer

    def set_default_decoder_optimizer(self, model: BaseAE) -> torch.optim.Optimizer:

        optimizer = optim.Adam(
            model.decoder.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.decoder_optim_decay,
        )

        return optimizer

    def set_default_discriminator_optimizer(
        self, model: BaseAE
    ) -> torch.optim.Optimizer:

        optimizer = optim.Adam(
            model.discriminator.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.discriminator_optim_decay,
        )

        return optimizer

    def _reset_optimizers_grads(self):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def _optimizers_step(self, model_output):

        encoder_loss = model_output.encoder_loss
        decoder_loss = model_output.decoder_loss
        discriminator_loss = model_output.discriminator_loss

        # Reset optimizers
        if model_output.update_encoder:
            encoder_loss.backward(retain_graph=True)

        if model_output.update_decoder:
            decoder_loss.backward(retain_graph=True)

        if model_output.update_discriminator:
            discriminator_loss.backward()

        if model_output.update_encoder:
            self.encoder_optimizer.step()

        if model_output.update_decoder:
            self.decoder_optimizer.step()

        if model_output.update_discriminator:
            self.discriminator_optimizer.step()

    def _schedulers_step(
        self, encoder_metrics=None, decoder_metrics=None, discriminator_metrics=None
    ):
        if isinstance(self.encoder_scheduler, ReduceLROnPlateau):
            self.encoder_scheduler.step(encoder_metrics)

        else:
            self.encoder_scheduler.step()

        if isinstance(self.decoder_scheduler, ReduceLROnPlateau):
            self.decoder_scheduler.step(decoder_metrics)

        else:
            self.decoder_scheduler.step()

        if isinstance(self.discriminator_scheduler, ReduceLROnPlateau):
            self.discriminator_scheduler.step(discriminator_metrics)

        else:
            self.discriminator_scheduler.step()

    def train(self, log_output_dir: str = None):
        """This function is the main training function

        Args:
            log_output_dir (str): The path in which the log will be stored
        """

        self.callback_handler.on_train_begin(
            training_config=self.training_config, model_config=self.model.model_config
        )

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

        for epoch in range(1, self.training_config.num_epochs + 1):

            self.callback_handler.on_epoch_begin(
                training_config=self.training_config,
                epoch=epoch,
                train_loader=self.train_loader,
                eval_loader=self.eval_loader,
            )

            metrics = {}

            train_losses = self.train_step(epoch)

            [
                epoch_train_loss,
                epoch_train_encoder_loss,
                epoch_train_decoder_loss,
                epoch_train_discriminator_loss,
            ] = train_losses
            metrics["train_epoch_loss"] = epoch_train_loss
            metrics["train_encoder_loss"] = epoch_train_encoder_loss
            metrics["train_decoder_loss"] = epoch_train_decoder_loss
            metrics["train_discriminator_loss"] = epoch_train_discriminator_loss

            if self.eval_dataset is not None:
                eval_losses = self.eval_step(epoch)

                [
                    epoch_eval_loss,
                    epoch_eval_encoder_loss,
                    epoch_eval_decoder_loss,
                    epoch_eval_discriminator_loss,
                ] = eval_losses
                metrics["eval_epoch_loss"] = epoch_eval_loss
                metrics["eval_encoder_loss"] = epoch_eval_encoder_loss
                metrics["eval_decoder_loss"] = epoch_eval_decoder_loss
                metrics["eval_discriminator_loss"] = epoch_eval_discriminator_loss

                self._schedulers_step(
                    encoder_metrics=epoch_eval_encoder_loss,
                    decoder_metrics=epoch_eval_decoder_loss,
                    discriminator_metrics=epoch_eval_discriminator_loss,
                )

            else:
                epoch_eval_loss = best_eval_loss
                self._schedulers_step(
                    encoder_metrics=epoch_train_encoder_loss,
                    decoder_metrics=epoch_train_decoder_loss,
                    discriminator_metrics=epoch_train_discriminator_loss,
                )

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
                self.save_checkpoint(
                    model=best_model, dir_path=training_dir, epoch=epoch
                )
                logger.info(f"Saved checkpoint at epoch {epoch}\n")

                if log_verbose:
                    file_logger.info(f"Saved checkpoint at epoch {epoch}\n")

            self.callback_handler.on_log(
                self.training_config, metrics, logger=logger, global_step=epoch
            )

        final_dir = os.path.join(training_dir, "final_model")

        self.save_model(best_model, dir_path=final_dir)
        logger.info("----------------------------------")
        logger.info("Training ended!")
        logger.info(f"Saved final model in {final_dir}")

        self.callback_handler.on_train_end(training_config=self.training_config)

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

        for inputs in self.eval_loader:

            self.callback_handler.on_eval_step_begin(
                training_config=self.training_config,
                eval_loader=self.eval_loader,
                epoch=epoch,
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

            self.callback_handler.on_eval_step_end(training_config=self.training_config)

        epoch_encoder_loss /= len(self.eval_loader)
        epoch_decoder_loss /= len(self.eval_loader)
        epoch_discriminator_loss /= len(self.eval_loader)
        epoch_loss /= len(self.eval_loader)

        return (
            epoch_loss,
            epoch_encoder_loss,
            epoch_decoder_loss,
            epoch_discriminator_loss,
        )

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

        for inputs in self.train_loader:

            self.callback_handler.on_train_step_begin(
                training_config=self.training_config,
                train_loader=self.train_loader,
                epoch=epoch,
            )

            inputs = self._set_inputs_to_device(inputs)

            self._reset_optimizers_grads()

            model_output = self.model(
                inputs, epoch=epoch, dataset_size=len(self.train_loader.dataset)
            )

            self._optimizers_step(model_output)

            encoder_loss = model_output.encoder_loss
            decoder_loss = model_output.decoder_loss
            discriminator_loss = model_output.discriminator_loss

            loss = encoder_loss + decoder_loss + discriminator_loss

            epoch_encoder_loss += encoder_loss.item()
            epoch_decoder_loss += decoder_loss.item()
            epoch_discriminator_loss += discriminator_loss.item()
            epoch_loss += loss.item()

            self.callback_handler.on_train_step_end(
                training_config=self.training_config
            )

        # Allows model updates if needed
        self.model.update()

        epoch_encoder_loss /= len(self.train_loader)
        epoch_decoder_loss /= len(self.train_loader)
        epoch_discriminator_loss /= len(self.train_loader)
        epoch_loss /= len(self.train_loader)

        return (
            epoch_loss,
            epoch_encoder_loss,
            epoch_decoder_loss,
            epoch_discriminator_loss,
        )

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
