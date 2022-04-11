import logging
from typing import List, Optional, Union

import numpy as np
import torch
from torch.optim import Optimizer

from ..customexception import LoadError
from ..data.preprocessors import DataProcessor
from ..models import VAE, BaseAE, VAEConfig
from ..trainers import *
from ..trainers.training_callbacks import TrainingCallback
from .base_pipeline import Pipeline

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class TrainingPipeline(Pipeline):
    """
    This Pipeline provides an end to end way to train your VAE model.
    The trained model will be saved in ``output_dir`` stated in the
    :class:`~pythae.trainers.BaseTrainerConfig`. A folder
    ``training_YYYY-MM-DD_hh-mm-ss`` is
    created where checkpoints and final model will be saved. Checkpoints are saved in
    ``checkpoint_epoch_{epoch}`` folder (optimizer and training config
    saved as well to resume training if needed)
    and the final model is saved in a ``final_model`` folder. If ``output_dir`` is
    None, data is saved in ``dummy_output_dir/training_YYYY-MM-DD_hh-mm-ss`` is created.

    Parameters:

        model (Optional[BaseAE]): An instance of :class:`~pythae.models.BaseAE` you want to train.
            If None, a default :class:`~pythae.models.VAE` model is used. Default: None.

        training_config (Optional[BaseTrainerConfig]): An instance of
            :class:`~pythae.trainers.BaseTrainerConfig` stating the training
            parameters. If None, a default configuration is used.
    """

    def __init__(
        self,
        model: Optional[BaseAE] = None,
        training_config: Optional[BaseTrainerConfig] = None,
    ):

        if model is not None:
            if training_config is None:
                if model.model_name == "RAE_L2":
                    training_config = CoupledOptimizerTrainerConfig(
                        encoder_optim_decay=0,
                        decoder_optim_decay=model.model_config.reg_weight,
                    )

                elif (
                    model.model_name == "Adversarial_AE"
                    or model.model_name == "FactorVAE"
                ):
                    training_config = AdversarialTrainerConfig()

                elif model.model_name == "VAEGAN":
                    training_config = CoupledOptimizerAdversarialTrainerConfig()

                else:
                    training_config = BaseTrainerConfig()

            elif model.model_name == "RAE_L2":
                if not isinstance(training_config, CoupledOptimizerTrainerConfig):

                    raise AssertionError(
                        "A 'CoupledOptimizerTrainerConfig' "
                        "is expected for training a RAE_L2"
                    )

                training_config.encoder_optim_decay = 0.0
                training_config.decoder_optim_decay = model.model_config.reg_weight

            elif (
                model.model_name == "Adversarial_AE" or model.model_name == "FactorVAE"
            ):
                if not isinstance(training_config, AdversarialTrainerConfig):

                    raise AssertionError(
                        "A 'AdversarialTrainer' "
                        f"is expected for training a {model.model_name}"
                    )

            elif model.model_name == "VAEGAN":
                if not isinstance(
                    training_config, CoupledOptimizerAdversarialTrainerConfig
                ):

                    raise AssertionError(
                        "A 'CoupledOptimizerAdversarialTrainer' "
                        "is expected for training a VAEGAN"
                    )

            if not isinstance(training_config, BaseTrainerConfig):
                raise AssertionError(
                    "A 'BaseTrainerConfig' " "is expected for the pipeline"
                )

        else:
            training_config = BaseTrainerConfig()

        self.data_processor = DataProcessor()
        self.model = model
        self.training_config = training_config

    def _set_default_model(self, data):
        model_config = VAEConfig(input_dim=data.shape[1:])
        model = VAE(model_config)
        self.model = model

    def __call__(
        self,
        train_data: Union[np.ndarray, torch.Tensor],
        eval_data: Union[np.ndarray, torch.Tensor] = None,
        callbacks: List[TrainingCallback] = None,
    ):
        """
        Launch the model training on the provided data.

        Args:
            training_data (Union[~numpy.ndarray, ~torch.Tensor]): The training data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...)

            eval_data (Optional[Union[~numpy.ndarray, ~torch.Tensor]]): The evaluation data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...). If None, only uses train_fata for training. Default: None.

            callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
                A list of callbacks to use during training.
        """

        logger.info("Preprocessing train data...")
        train_data = self.data_processor.process_data(train_data)
        train_dataset = self.data_processor.to_dataset(train_data)

        self.train_data = train_data

        if self.model is None:
            self._set_default_model(train_data)

        if eval_data is not None:
            logger.info("Preprocessing eval data...\n")
            eval_data = self.data_processor.process_data(eval_data)
            eval_dataset = self.data_processor.to_dataset(eval_data)

        else:
            eval_dataset = None

        if isinstance(self.training_config, CoupledOptimizerTrainerConfig):
            logger.info("Using Coupled Optimizer Trainer\n")
            trainer = CoupledOptimizerTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, AdversarialTrainerConfig):
            logger.info("Using Adversarial Trainer\n")
            trainer = AdversarialTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, CoupledOptimizerAdversarialTrainerConfig):
            logger.info("Using Coupled Optimizer Adversarial Trainer\n")
            trainer = CoupledOptimizerAdversarialTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, BaseTrainerConfig):
            logger.info("Using Base Trainer\n")
            trainer = BaseTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        self.trainer = trainer

        trainer.train()
