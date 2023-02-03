import datetime
import logging
import os
from copy import deepcopy
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ...data.datasets import BaseDataset
from ...models import BaseAE
from ..base_trainer import BaseTrainer
from ..trainer_utils import set_seed
from ..training_callbacks import TrainingCallback
from .coupled_optimizer_trainer_config import CoupledOptimizerTrainerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class CoupledOptimizerTrainer(BaseTrainer):
    """Trainer using distinct optimizers for encoder and decoder nn.

    Args:
        model (BaseAE): The model to train

        train_dataset (BaseDataset): The training dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        training_args (CoupledOptimizerTrainerConfig): The training arguments summarizing the main
            parameters used for training. If None, a basic training instance of
            :class:`CoupledOptimizerTrainerConfig` is used. Default: None.

        encoder_optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer` used for
            training the encoder. If None, a :class:`~torch.optim.Adam` optimizer is used.
            Default: None.

        decoder_optimizer (~torch.optim.Optimizer): An instance of `torch.optim.Optimizer` used for
            training the decoder. If None, a :class:`~torch.optim.Adam` optimizer is used.
            Default: None.
    """

    def __init__(
        self,
        model: BaseAE,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
        training_config: Optional[CoupledOptimizerTrainerConfig] = None,
        callbacks: List[TrainingCallback] = None,
    ):

        BaseTrainer.__init__(
            self,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
            callbacks=callbacks,
        )

    def set_encoder_optimizer(self):
        encoder_optimizer_cls = getattr(
            optim, self.training_config.encoder_optimizer_cls
        )

        if self.training_config.encoder_optimizer_params is not None:
            if self.distributed:
                encoder_optimizer = encoder_optimizer_cls(
                    self.model.module.encoder.parameters(),
                    lr=self.training_config.encoder_learning_rate,
                    **self.training_config.encoder_optimizer_params,
                )
            else:
                encoder_optimizer = encoder_optimizer_cls(
                    self.model.encoder.parameters(),
                    lr=self.training_config.encoder_learning_rate,
                    **self.training_config.encoder_optimizer_params,
                )
        else:
            if self.distributed:
                encoder_optimizer = encoder_optimizer_cls(
                    self.model.module.encoder.parameters(),
                    lr=self.training_config.encoder_learning_rate,
                )
            else:
                encoder_optimizer = encoder_optimizer_cls(
                    self.model.encoder.parameters(),
                    lr=self.training_config.encoder_learning_rate,
                )

        self.encoder_optimizer = encoder_optimizer

    def set_encoder_scheduler(self):
        if self.training_config.encoder_scheduler_cls is not None:
            encoder_scheduler_cls = getattr(
                lr_scheduler, self.training_config.encoder_scheduler_cls
            )

            if self.training_config.encoder_scheduler_params is not None:
                scheduler = encoder_scheduler_cls(
                    self.encoder_optimizer,
                    **self.training_config.encoder_scheduler_params,
                )
            else:
                scheduler = encoder_scheduler_cls(self.encoder_optimizer)

        else:
            scheduler = None

        self.encoder_scheduler = scheduler

    def set_decoder_optimizer(self):
        decoder_cls = getattr(optim, self.training_config.decoder_optimizer_cls)

        if self.training_config.decoder_optimizer_params is not None:
            if self.distributed:
                decoder_optimizer = decoder_cls(
                    self.model.module.decoder.parameters(),
                    lr=self.training_config.decoder_learning_rate,
                    **self.training_config.decoder_optimizer_params,
                )
            else:
                decoder_optimizer = decoder_cls(
                    self.model.decoder.parameters(),
                    lr=self.training_config.decoder_learning_rate,
                    **self.training_config.decoder_optimizer_params,
                )

        else:
            if self.distributed:
                decoder_optimizer = decoder_cls(
                    self.model.module.decoder.parameters(),
                    lr=self.training_config.decoder_learning_rate,
                )
            else:
                decoder_optimizer = decoder_cls(
                    self.model.decoder.parameters(),
                    lr=self.training_config.decoder_learning_rate,
                )

        self.decoder_optimizer = decoder_optimizer

    def set_decoder_scheduler(self) -> torch.optim.lr_scheduler:
        if self.training_config.decoder_scheduler_cls is not None:
            decoder_scheduler_cls = getattr(
                lr_scheduler, self.training_config.decoder_scheduler_cls
            )

            if self.training_config.decoder_scheduler_params is not None:
                scheduler = decoder_scheduler_cls(
                    self.decoder_optimizer,
                    **self.training_config.decoder_scheduler_params,
                )
            else:
                scheduler = decoder_scheduler_cls(self.decoder_optimizer)

        else:
            scheduler = None

        self.decoder_scheduler = scheduler

    def _optimizers_step(self, model_output):

        encoder_loss = model_output.encoder_loss
        decoder_loss = model_output.decoder_loss

        # Reset optimizers
        if model_output.update_encoder:
            self.encoder_optimizer.zero_grad()
            encoder_loss.backward(retain_graph=True)

        if model_output.update_decoder:
            self.decoder_optimizer.zero_grad()
            decoder_loss.backward(retain_graph=True)

        if model_output.update_encoder:
            self.encoder_optimizer.step()

        if model_output.update_decoder:
            self.decoder_optimizer.step()

    def _schedulers_step(self, encoder_metrics=None, decoder_metrics=None):

        if self.encoder_scheduler is None:
            pass

        elif isinstance(self.encoder_scheduler, lr_scheduler.ReduceLROnPlateau):
            self.encoder_scheduler.step(encoder_metrics)

        else:
            self.encoder_scheduler.step()

        if self.decoder_scheduler is None:
            pass

        elif isinstance(self.decoder_scheduler, lr_scheduler.ReduceLROnPlateau):
            self.decoder_scheduler.step(decoder_metrics)

        else:
            self.decoder_scheduler.step()

    def prepare_training(self):

        # set random seed
        set_seed(self.training_config.seed)

        # set autoencoder optimizer and scheduler
        self.set_encoder_optimizer()
        self.set_encoder_scheduler()

        # set discriminator optimizer and scheduler
        self.set_decoder_optimizer()
        self.set_decoder_scheduler()

        # create foder for saving
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
            f"Encoder Optimizer: {self.encoder_optimizer}\n"
            f"Encoder Scheduler: {self.encoder_scheduler}\n"
            f"Decoder Optimizer: {self.decoder_optimizer}\n"
            f"Decoder Scheduler: {self.decoder_scheduler}\n"
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

            train_losses = self.train_step(epoch)

            [
                epoch_train_loss,
                epoch_train_encoder_loss,
                epoch_train_decoder_loss,
            ] = train_losses
            metrics["train_epoch_loss"] = epoch_train_loss
            metrics["train_encoder_loss"] = epoch_train_encoder_loss
            metrics["train_decoder_loss"] = epoch_train_decoder_loss

            if self.eval_dataset is not None:
                eval_losses = self.eval_step(epoch)

                [
                    epoch_eval_loss,
                    epoch_eval_encoder_loss,
                    epoch_eval_decoder_loss,
                ] = eval_losses
                metrics["eval_epoch_loss"] = epoch_eval_loss
                metrics["eval_encoder_loss"] = epoch_eval_encoder_loss
                metrics["eval_decoder_loss"] = epoch_eval_decoder_loss

                self._schedulers_step(
                    encoder_metrics=epoch_eval_encoder_loss,
                    decoder_metrics=epoch_eval_decoder_loss,
                )

            else:
                epoch_eval_loss = best_eval_loss
                self._schedulers_step(
                    encoder_metrics=epoch_train_encoder_loss,
                    decoder_metrics=epoch_train_decoder_loss,
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
            logger.info("----------------------------------")
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

        epoch_encoder_loss = 0
        epoch_decoder_loss = 0
        epoch_loss = 0

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

            encoder_loss = model_output.encoder_loss
            decoder_loss = model_output.decoder_loss

            loss = encoder_loss + decoder_loss

            epoch_encoder_loss += encoder_loss.item()
            epoch_decoder_loss += decoder_loss.item()
            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in eval loss")

            self.callback_handler.on_eval_step_end(training_config=self.training_config)

        epoch_encoder_loss /= len(self.eval_loader)
        epoch_decoder_loss /= len(self.eval_loader)
        epoch_loss /= len(self.eval_loader)

        return (epoch_loss, epoch_encoder_loss, epoch_decoder_loss)

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

        epoch_encoder_loss = 0
        epoch_decoder_loss = 0
        epoch_loss = 0

        for inputs in self.train_loader:

            inputs = self._set_inputs_to_device(inputs)

            model_output = self.model(
                inputs,
                epoch=epoch,
                dataset_size=len(self.train_loader.dataset),
                uses_ddp=self.distributed,
            )

            self._optimizers_step(model_output)

            encoder_loss = model_output.encoder_loss
            decoder_loss = model_output.decoder_loss

            loss = encoder_loss + decoder_loss

            epoch_encoder_loss += encoder_loss.item()
            epoch_decoder_loss += decoder_loss.item()
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

        epoch_encoder_loss /= len(self.train_loader)
        epoch_decoder_loss /= len(self.train_loader)
        epoch_loss /= len(self.train_loader)

        return (epoch_loss, epoch_encoder_loss, epoch_decoder_loss)

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

        # save model
        if self.distributed:
            model.module.save(checkpoint_dir)

        else:
            model.save(checkpoint_dir)

        # save training config
        self.training_config.save_json(checkpoint_dir, "training_config")
