import logging
from typing import Optional, Union

import numpy as np
import torch

from ..models import BaseAE
from ..samplers import *
from ..trainers import BaseTrainerConfig
from .base_pipeline import Pipeline

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class GenerationPipeline(Pipeline):
    """
    This Pipeline provides an end to end way to generate samples from a trained VAE model. It only
    needs a :class:`pythae.models` to sample from and a smapler configuration.

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
        sampler_config: Optional[BaseSamplerConfig] = None,
    ):

        if sampler_config is None:
            sampler_config = NormalSamplerConfig()

        if sampler_config.name == "NormalSamplerConfig":
            sampler = NormalSampler(model=model, sampler_config=sampler_config)

        elif sampler_config.name == "GaussianMixtureSamplerConfig":
            sampler = GaussianMixtureSampler(model=model, sampler_config=sampler_config)

        elif sampler_config.name == "IAFSamplerConfig":
            sampler = IAFSampler(model=model, sampler_config=sampler_config)

        elif sampler_config.name == "MAFSamplerConfig":
            sampler = MAFSampler(model=model, sampler_config=sampler_config)

        elif sampler_config.name == "RHVAESamplerConfig":
            sampler = RHVAESampler(model=model, sampler_config=sampler_config)

        elif sampler_config.name == "PixelCNNSamplerConfig":
            sampler = PixelCNNSampler(model=model, sampler_config=sampler_config)

        elif sampler_config.name == "TwoStageVAESamplerConfig":
            sampler = TwoStageVAESampler(model=model, sampler_config=sampler_config)

        elif sampler_config.name == "VAMPSamplerConfig":
            sampler = VAMPSampler(model=model, sampler_config=sampler_config)

        elif sampler_config.name == "HypersphereUniformSamplerConfig":
            sampler = HypersphereUniformSampler(
                model=model, sampler_config=sampler_config
            )

        elif sampler_config.name == "PoincareDiskSamplerConfig":
            sampler = PoincareDiskSampler(model=model, sampler_config=sampler_config)

        else:
            raise NotImplementedError(
                "Unrecognized sampler config name... Check that that the sampler_config name "
                f"is written correctly. Got '{sampler_config.name}'."
            )

        self.sampler = sampler

    def __call__(
        self,
        num_samples: int = 1,
        batch_size: int = 500,
        output_dir: str = None,
        return_gen: bool = True,
        save_sampler_config: bool = False,
        train_data: Union[np.ndarray, torch.Tensor] = None,
        eval_data: Union[np.ndarray, torch.Tensor] = None,
        training_config: BaseTrainerConfig = None,
    ):
        """
        Launch the model training on the provided data.

        Args:
            num_samples (int): The number of samples to generate

            batch_size (int): The batch size to use during sampling

            output_dir (str): The directory where the images will be saved. If does not exist the
                folder is created. If None: the images are not saved. Defaults: None.

            return_gen (bool): Whether the sampler should directly return a tensor of generated
                data. Default: True.

            training_data (Union[~numpy.ndarray, ~torch.Tensor]): The training data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...) if the sampler needs to be trained (e.g. flow based samplers).
                Default: None.

            eval_data (Optional[Union[~numpy.ndarray, ~torch.Tensor]]): The evaluation data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...) if the sampler needs to be trained (e.g. flow based samplers).
                Default: None.

            training_config (BaseTrainerConfig): the training config to use if the sampler needs to
                be trained (e.g. flow based samplers). Default: None.

            callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
                A list of callbacks to use during training.
        """

        # Fit the sampler
        self.sampler.fit(
            train_data=train_data, eval_data=eval_data, training_config=training_config
        )

        # Generate data
        generated_samples = self.sampler.sample(
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=output_dir,
            return_gen=return_gen,
            save_sampler_config=save_sampler_config,
        )

        return generated_samples
