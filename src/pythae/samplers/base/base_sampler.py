import logging
import os

import torch

from .base_sampler_config import BaseSamplerConfig
from ...models import BaseAE

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseSampler:
    """Base class for sampler used to generate from the VAEs models

    Args:
        model (BaseAE): The vae model to sample from.
        sampler_config (BaseSamplerConfig): An instance of BaseSamplerConfig in which any sampler's
            parameters is made available. If None a default configuration is used. Default: None
    """

    def __init__(self, model: BaseAE, sampler_config: BaseSamplerConfig = None):

        if sampler_config.output_dir is None:
            output_dir = "dummy_output_dir"
            sampler_config.output_dir = output_dir

        if not os.path.exists(sampler_config.output_dir):
            os.makedirs(sampler_config.output_dir)
            logger.info(
                f"Created {sampler_config.output_dir} folder since did not exist.\n"
            )

        self.model = model
        self.sampler_config = sampler_config

        self.batch_size = sampler_config.batch_size
        self.samples_per_save = self.sampler_config.samples_per_save

        self.device = (
            "cuda"
            if torch.cuda.is_available() and not sampler_config.no_cuda
            else "cpu"
        )

        self.model.to(self.device)

    def fit(self, *args, **kwargs):
        """Function to be called to fit the sampler before sampling
        """
        pass

    def sample(self, num_samples):
        """Main sampling function of the samplers. The data is saved in the
        ``output_dir/generation_``
        folder passed in the `~pythae.models.model_config.SamplerConfig` instance. If ``output_dir``
        if None, a folder named ``dummy_output_dir`` is created in this folder.

        Args:
            num_samples (int): The number of samples to generate
        """
        raise NotImplementedError()

    def save(self, dir_path):
        """Method to save the sampler config. The config is saved a as ``sampler_config.json``
        file in ``dir_path``"""

        self.sampler_config.save_json(dir_path, "sampler_config")

    def save_data_batch(self, data, dir_path, number_of_samples, batch_idx):
        """
        Method to save a batch of generated data. The data will be saved in the
        ``dir_path`` folder. The batch of data
        is saved in a file named ``generated_data_{number_of_samples}_{batch_idx}.pt``

        Args:
            data (torch.Tensor): The data to save
            dir_path (str): The folder where the data and config file must be saved
            batch_idx (int): The batch idx

        .. note::
            You can then easily reload the generated data using

            .. code-block:

                >>> import torch
                >>> import os
                >>> data = torch.load(
                ...    os.path.join(
                ...        'dir_path', 'generated_data_{number_of_samples}_{batch_idx}.pt'))
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        torch.save(
            data,
            os.path.join(
                dir_path, f"generated_data_{number_of_samples}_{batch_idx}.pt"
            ),
        )
