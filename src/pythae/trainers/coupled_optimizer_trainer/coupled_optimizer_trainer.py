import datetime
import logging
import os
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
from ..base_trainer import BaseTrainer
from .coupled_optimizer_trainer_config import CoupledOptimizerTrainerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class CoupledOptimizerTrainer(BaseTrainer):
    """Trainer using disctinct optimizers for encoder and decoder nn.

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
        encoder_optimizer: Optional[torch.optim.Optimizer] = None,
        decoder_optimizer: Optional[torch.optim.Optimizer] = None
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
            encoder_optimizer = self._set_optimizer_on_device(encoder_optimizer, self.device)

        # set decoder optimizer
        if decoder_optimizer is None:
            decoder_optimizer = self.set_default_decoder_optimizer(model)

        else:
            decoder_optimizer = self._set_optimizer_on_device(decoder_optimizer, self.device)

        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

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

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                model_output = self.model(inputs)

                loss = model_output.loss

                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                epoch_loss += loss.item()

            # Allows model updates if needed
            self.model.update()

            epoch_loss /= len(self.train_loader)

        return epoch_loss


    def save_checkpoint(self, dir_path, epoch: int):
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
        self.model.save(checkpoint_dir)

        # save training config
        self.training_config.save_json(checkpoint_dir, "training_config")
