import logging
from typing import List, Optional, Union

import numpy as np
import torch

from ..customexception import DatasetError
from ..data.datasets import collate_dataset_output
from ..data.preprocessors import BaseDataset, DataProcessor
from ..models import BaseAE
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
        model: Optional[BaseAE],
        training_config: Optional[BaseTrainerConfig] = None,
    ):

        if training_config is None:
            if model.model_name == "RAE_L2":
                training_config = CoupledOptimizerTrainerConfig(
                    encoder_optimizer_params={"weight_decay": 0},
                    decoder_optimizer_params={"weight_decay": model.model_config.reg_weight},
                )

            elif (
                model.model_name == "Adversarial_AE" or model.model_name == "FactorVAE"
            ):
                training_config = AdversarialTrainerConfig()

            elif model.model_name == "VAEGAN":
                training_config = CoupledOptimizerAdversarialTrainerConfig()

            else:
                training_config = BaseTrainerConfig()

        elif model.model_name == "RAE_L2" or model.model_name == "PIWAE":
            if not isinstance(training_config, CoupledOptimizerTrainerConfig):

                raise AssertionError(
                    "A 'CoupledOptimizerTrainerConfig' "
                    f"is expected for training a {model.model_name}"
                )
            if model.model_name == "RAE_L2":
                if training_config.decoder_optimizer_params is None:
                    training_config.decoder_optimizer_params = {
                        "weight_decay": model.model_config.reg_weight
                    }
                else:
                    training_config.decoder_optimizer_params[
                        "weight_decay"
                    ] = model.model_config.reg_weight

        elif model.model_name == "Adversarial_AE" or model.model_name == "FactorVAE":
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

        self.data_processor = DataProcessor()
        self.model = model
        self.training_config = training_config

    def _check_dataset(self, dataset: BaseDataset):

        try:
            dataset_output = dataset[0]

        except Exception as e:
            raise DatasetError(
                "Error when trying to collect data from the dataset. Check `__getitem__` method. "
                "The Dataset should output a dictionnary with keys at least ['data']. "
                "Please check documentation.\n"
                f"Exception raised: {type(e)} with message: " + str(e)
            ) from e

        if "data" not in dataset_output.keys():
            raise DatasetError(
                "The Dataset should output a dictionnary with keys ['data']"
            )

        try:
            len(dataset)

        except Exception as e:
            raise DatasetError(
                "Error when trying to get dataset len. Check `__len__` method. "
                "Please check documentation.\n"
                f"Exception raised: {type(e)} with message: " + str(e)
            ) from e

        # check everything if fine when combined with data loader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=min(len(dataset), 2),
            collate_fn=collate_dataset_output,
        )
        loader_out = next(iter(dataloader))
        assert loader_out.data.shape[0] == min(
            len(dataset), 2
        ), "Error when combining dataset with loader."

    def __call__(
        self,
        train_data: Union[
            np.ndarray,
            torch.Tensor,
            torch.utils.data.Dataset,
            torch.utils.data.DataLoader,
        ] = None,
        eval_data: Union[
            np.ndarray,
            torch.Tensor,
            torch.utils.data.Dataset,
            torch.utils.data.DataLoader,
        ] = None,
        callbacks: List[TrainingCallback] = None,
    ):
        """
        Launch the model training on the provided data.

        Args:
            train_data: The training data or DataLoader.

            eval_data: The evaluation data or DataLoader. If None, only uses train_data for training. Default: None

            callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
                A list of callbacks to use during training.
        """

        # Initialize variables for datasets and dataloaders
        train_dataset, eval_dataset = None, None
        train_dataloader, eval_dataloader = None, None

        if isinstance(train_data, torch.utils.data.DataLoader):
            train_dataloader = train_data
        elif isinstance(train_data, (np.ndarray, torch.Tensor)):
            logger.info("Preprocessing train data...")
            train_data = self.data_processor.process_data(train_data)
            train_dataset = self.data_processor.to_dataset(train_data)
            logger.info("Checking train dataset...")
            self._check_dataset(train_dataset)
        else:
            train_dataset = train_data
            logger.info("Checking train dataset...")
            self._check_dataset(train_dataset)

        if eval_data is not None:
            if isinstance(eval_data, torch.utils.data.DataLoader):
                eval_dataloader = eval_data
            elif isinstance(eval_data, (np.ndarray, torch.Tensor)):
                logger.info("Preprocessing eval data...\n")
                eval_data = self.data_processor.process_data(eval_data)
                eval_dataset = self.data_processor.to_dataset(eval_data)
                logger.info("Checking eval dataset...")
                self._check_dataset(eval_dataset)
            else:
                eval_dataset = eval_data
                logger.info("Checking eval dataset...")
                self._check_dataset(eval_dataset)

        if isinstance(self.training_config, CoupledOptimizerTrainerConfig):
            logger.info("Using Coupled Optimizer Trainer\n")
            trainer = CoupledOptimizerTrainer(
                model=self.model,
                train_dataset=train_dataloader or train_dataset,
                eval_dataset=eval_dataloader or eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, AdversarialTrainerConfig):
            logger.info("Using Adversarial Trainer\n")
            trainer = AdversarialTrainer(
                model=self.model,
                train_dataset=train_dataloader or train_dataset,
                eval_dataset=eval_dataloader or eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, CoupledOptimizerAdversarialTrainerConfig):
            logger.info("Using Coupled Optimizer Adversarial Trainer\n")
            trainer = CoupledOptimizerAdversarialTrainer(
                model=self.model,
                train_dataset=train_dataloader or train_dataset,
                eval_dataset=eval_dataloader or eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, BaseTrainerConfig):
            logger.info("Using Base Trainer\n")
            trainer = BaseTrainer(
                model=self.model,
                train_dataset=train_dataloader or train_dataset,
                eval_dataset=eval_dataloader or eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )
        else:
            raise ValueError("The provided training config is not supported.")

        self.trainer = trainer
        trainer.train()
