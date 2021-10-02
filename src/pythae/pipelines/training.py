from typing import Optional, Union

import numpy as np
import torch
from torch.optim import Optimizer

from pyraug.customexception import LoadError
from pyraug.data.loaders import BaseDataGetter, ImageGetterFromFolder
from pyraug.data.preprocessors import DataProcessor
from pyraug.models import RHVAE, BaseVAE
from pyraug.models.rhvae import RHVAEConfig
from pyraug.trainers import Trainer
from pyraug.trainers.training_config import TrainingConfig

from .base_pipeline import Pipeline


class TrainingPipeline(Pipeline):
    """
    This Pipeline provides an end to end way to train your VAE model.
    The trained model will be saved in ``output_dir`` stated in the
    :class:`~pyraug.trainers.training_config.TrainingConfig`. A folder
    ``training_YYYY-MM-DD_hh-mm-ss`` is
    created where checkpoints and final model will be saved. Checkpoints are saved in
    ``checkpoint_epoch_{epoch}`` folder (optimizer and training config
    saved as well to resume training if needed)
    and the final model is saved in a ``final_model`` folder. If ``output_dir`` is
    None, data is saved in ``dummy_output_dir/training_YYYY-MM-DD_hh-mm-ss`` is created.

    Parameters:

        data_loader (Optional[BaseDataGetter]): The data loader you want to use to load your
            data. This is usefull to get the data from a particular format and in a specific folder
            for instance. If None, the :class:`~pyraug.data.loaders.ImageGetterFromFolder` is used.
            Default: None.

        data_processor (Optional[DataProcessor]): The data preprocessor you want to use to
            preprocess your data (*e.g.* normalization, reshaping, type conversion). If None,
            a basic :class:`~pyraug.data.preprocessors.DataProcessor` is used (by default data
            is normalized such that the max value of each data is 1 and the min 0). Default: None.

        model (Optional[BaseVAE]): An instance of :class:`~pyraug.models.BaseVAE` you want to train.
            If None, a default :class:`~pyraug.models.RHVAE` model is used. Default: None.

        optimizer (Optional[~torch.optim.Optimizer]): An instance of :class:`~torch.optim.Optimizer`
            used to train the model. If None we provide an instance of
            :class:`~torch.optim.Adam` optimizer. Default: None.

        training_config (Optional[TrainingConfig]=None): An instance of
            :class:`~pyraug.trainers.training_config.TrainingConfig` stating the training
            parameters. If None, a default configuration is used.

    .. note::
            If you did not provide any data_processor, a default one will be used. By default it
            normalizes the data so that the max value of each data equals 1 and min value 0.
    """

    def __init__(
        self,
        data_loader: Optional[BaseDataGetter] = None,
        data_processor: Optional[DataProcessor] = None,
        model: Optional[BaseVAE] = None,
        optimizer: Optional[Optimizer] = None,
        training_config: Optional[TrainingConfig] = None,
    ):

        # model_name = model_name.upper()

        self.data_loader = data_loader

        if data_processor is None:
            data_processor = DataProcessor(
                data_normalization_type="individual_min_max_scaling"
            )

        self.data_processor = data_processor
        self.model = model
        self.optimizer = optimizer
        self.training_config = training_config

    def _set_default_model(self, data):
        model_config = RHVAEConfig(input_dim=int(np.prod(data.shape[1:])))
        model = RHVAE(model_config)
        self.model = model

    def __call__(
        self,
        train_data: Union[str, np.ndarray, torch.Tensor],
        eval_data: Union[str, np.ndarray, torch.Tensor] = None,
        log_output_dir: str = None,
    ):
        """
        Launch the model training on the provided data.

        Args:
            training_data (Union[str, ~numpy.ndarray, ~torch.Tensor]): The training data coming from
                a folder in which each file is a data or a :class:`numpy.ndarray` or
                :class:`torch.Tensor` of shape (mini_batch x n_channels x data_shape)

            eval_data (Optional[Union[str, ~numpy.ndarray, ~torch.Tensor]]): The evaluation data coming from
                a folder in which each file is a data or a np.ndarray or torch.Tensor. If None, no
                evaluation data is used.

        """

        if self.data_loader is None:
            if isinstance(train_data, str):

                self.data_loader = ImageGetterFromFolder()

                try:
                    train_data = self.data_loader.load(train_data)

                except Exception as e:
                    raise LoadError(
                        f"Unable to load training data. Exception catch: {type(e)} with message: "
                        + str(e)
                    )

        else:
            try:
                train_data = self.data_loader.load(train_data)

            except Exception as e:
                raise LoadError(
                    f"Unable to load training data. Exception catch: {type(e)} with message: "
                    + str(e)
                )

        train_data = self.data_processor.process_data(train_data)
        train_dataset = self.data_processor.to_dataset(train_data)

        self.train_data = train_data

        if self.model is None:
            self._set_default_model(train_data)

        if eval_data is not None:
            if self.data_loader is None:
                if isinstance(eval_data, str):

                    self.data_loader = ImageGetterFromFolder()

                    try:
                        train_data = self.data_loader.load(eval_data)

                    except Exception as e:
                        raise LoadError(
                            f"Unable to load training data. Exception catch: {type(e)} with message: "
                            + str(e)
                        )

            else:
                try:
                    eval_data = self.data_loader.load(eval_data)

                except Exception as e:
                    raise LoadError(
                        f"Enable to load eval data. Exception catch: {type(e)} with message: "
                        + str(e)
                    )
            eval_data = self.data_processor.process_data(eval_data)
            eval_dataset = self.data_processor.to_dataset(eval_data)

        else:
            eval_dataset = None

        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=self.training_config,
            optimizer=self.optimizer,
        )

        self.trainer = trainer

        trainer.train(log_output_dir=log_output_dir)
