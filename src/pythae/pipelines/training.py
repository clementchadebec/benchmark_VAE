from typing import Optional, Union

import logging
import numpy as np
import torch
from torch.optim import Optimizer

from ..customexception import LoadError
from ..data.preprocessors import DataProcessor
from ..models import BaseAE, VAE, VAEConfig
from ..trainers import *

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
    :class:`~pythae.trainers.BaseTrainingConfig`. A folder
    ``training_YYYY-MM-DD_hh-mm-ss`` is
    created where checkpoints and final model will be saved. Checkpoints are saved in
    ``checkpoint_epoch_{epoch}`` folder (optimizer and training config
    saved as well to resume training if needed)
    and the final model is saved in a ``final_model`` folder. If ``output_dir`` is
    None, data is saved in ``dummy_output_dir/training_YYYY-MM-DD_hh-mm-ss`` is created.

    Parameters:

        data_loader (Optional[BaseDataGetter]): The data loader you want to use to load your
            data. This is usefull to get the data from a particular format and in a specific folder
            for instance. If None, the :class:`~pythae.data.loaders.ImageGetterFromFolder` is used.
            Default: None.

        data_processor (Optional[DataProcessor]): The data preprocessor you want to use to
            preprocess your data (*e.g.* normalization, reshaping, type conversion). If None,
            a basic :class:`~pythae.data.preprocessors.DataProcessor` is used (by default data
            is normalized such that the max value of each data is 1 and the min 0). Default: None.

        model (Optional[BaseAE]): An instance of :class:`~pythae.models.BaseAE` you want to train.
            If None, a default :class:`~pythae.models.VAE` model is used. Default: None.

        optimizer (Optional[~torch.optim.Optimizer]): An instance of :class:`~torch.optim.Optimizer`
            used to train the model. If None we provide an instance of
            :class:`~torch.optim.Adam` optimizer. Default: None.

        training_config (Optional[BaseTrainingConfig]=None): An instance of
            :class:`~pythae.trainers.BaseTrainingConfig` stating the training
            parameters. If None, a default configuration is used.

    .. note::
            If you did not provide any data_processor, a default one will be used. By default it
            normalizes the data so that the max value of each data equals 1 and min value 0.
    """

    def __init__(
        self,
        model: Optional[BaseAE] = None,
        training_config: Optional[BaseTrainingConfig] = None,
    ):

        if model.model_name == 'RAE_L2':
            if not isinstance(
                training_config, CoupledOptimizerTrainerConfig):

                raise AssertionError("A 'CoupledOptimizerTrainerConfig' "
                    "is expected for training a RAE_L2")


            training_config.encoder_optim_decay = 0.
            training_config.decoder_optim_decay = model.model_config.reg_weight

        self.data_processor = DataProcessor()
        self.model = model
        self.training_config = training_config

    def _set_default_model(self, data):
        model_config = VAEConfig(input_dim=int(np.prod(data.shape[1:])))
        model = VAE(model_config)
        self.model = model

    def __call__(
        self,
        train_data: Union[np.ndarray, torch.Tensor],
        eval_data: Union[np.ndarray, torch.Tensor] = None,
        log_output_dir: str = None,
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
        """

        logger.info("Preprocessing train data...")
        train_data = self.data_processor.process_data(train_data)
        train_dataset = self.data_processor.to_dataset(train_data)

        self.train_data = train_data

        if self.model is None:
            self._set_default_model(train_data)

        if eval_data is not None:
            logger.info("Preprocessing eval data...")
            eval_data = self.data_processor.process_data(eval_data)
            eval_dataset = self.data_processor.to_dataset(eval_data)

        else:
            eval_dataset = None


        if isinstance(self.training_config, CoupledOptimizerTrainerConfig):
            trainer = CoupledOptimizerTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config
            )

        elif isinstance(self.training_config, BaseTrainingConfig):

            trainer = BaseTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config
            )

        self.trainer = trainer

        trainer.train(log_output_dir=log_output_dir)
