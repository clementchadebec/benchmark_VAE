import logging
from typing import Union

import numpy as np
import torch
from sklearn import mixture
from torch.utils.data import DataLoader, Dataset

from ...data.datasets import collate_dataset_output
from ...data.preprocessors import DataProcessor
from ...models import BaseAE
from ..base import BaseSampler
from .gaussian_mixture_config import GaussianMixtureSamplerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class GaussianMixtureSampler(BaseSampler):
    """Fits a Gaussian Mixture in the Autoencoder's latent space.

    Args:
        model (BaseAE): The vae model to sample from.
        sampler_config (BaseSamplerConfig): An instance of BaseSamplerConfig in which any sampler's
            parameters is made available. If None a default configuration is used. Default: None.

    .. note::

        The method :class:`~pythae.samplers.GaussianMixtureSampler.fit` must be called to fit the sampler
        before sampling.

    """

    def __init__(
        self, model: BaseAE, sampler_config: GaussianMixtureSamplerConfig = None
    ):

        if sampler_config is None:
            sampler_config = GaussianMixtureSamplerConfig()

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

        self.n_components = sampler_config.n_components

    def fit(
        self,
        train_data: Union[torch.Tensor, np.ndarray, Dataset],
        batch_size: int = 64,
        **kwargs,
    ):
        """Method to fit the sampler from the training data

        Args:
            train_data (Union[torch.Tensor, np.ndarray, Dataset]): The train data needed to
                retrieve the training embeddings and fit the mixture in the latent space.
            batch_size (int): The batch size to use to retrieve the embeddings. Default: 64.
        """
        self.is_fitted = True

        if not isinstance(train_data, Dataset):
            data_processor = DataProcessor()
            train_data = data_processor.process_data(train_data)
            train_dataset = data_processor.to_dataset(train_data)

        else:
            train_dataset = train_data

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_dataset_output,
        )

        z = []
        try:
            with torch.no_grad():
                for _, inputs in enumerate(train_loader):
                    inputs = self._set_inputs_to_device(inputs)
                    z_ = self.model(inputs).z
                    z.append(z_)

        except RuntimeError:
            for _, inputs in enumerate(train_loader):
                inputs = self._set_inputs_to_device(inputs)
                z_ = self.model(inputs).z.detach()
                z.append(z_)

        z = torch.cat(z)

        if self.n_components > z.shape[0]:
            self.n_components = z.shape[0]
            logger.warning(
                f"Setting the number of component to {z.shape[0]} since"
                "n_components > n_samples when fitting the gmm"
            )

        gmm = mixture.GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            max_iter=2000,
            verbose=0,
            tol=1e-3,
        )
        gmm.fit(z.cpu().detach())

        self.gmm = gmm

    def sample(
        self,
        num_samples: int = 1,
        batch_size: int = 500,
        output_dir: str = None,
        return_gen: bool = True,
        save_sampler_config: bool = False,
    ) -> torch.Tensor:
        """Main sampling function of the sampler.

        Args:
            num_samples (int): The number of samples to generate
            batch_size (int): The batch size to use during sampling
            output_dir (str): The directory where the images will be saved. If does not exist the
                folder is created. If None: the images are not saved. Defaults: None.
            return_gen (bool): Whether the sampler should directly return a tensor of generated
                data. Default: True.
            save_sampler_config (bool): Whether to save the sampler config. It is saved in
                output_dir

        Returns:
            ~torch.Tensor: The generated images
        """

        if not self.is_fitted:
            raise ArithmeticError(
                "The sampler needs to be fitted by calling smapler.fit() method"
                "before sampling."
            )

        full_batch_nbr = int(num_samples / batch_size)
        last_batch_samples_nbr = num_samples % batch_size

        x_gen_list = []

        for i in range(full_batch_nbr):

            z = (
                torch.tensor(self.gmm.sample(batch_size)[0])
                .to(self.device)
                .type(torch.float)
            )
            x_gen = self.model.decoder(z)["reconstruction"].detach()

            if output_dir is not None:
                for j in range(batch_size):
                    self.save_img(
                        x_gen[j], output_dir, "%08d.png" % int(batch_size * i + j)
                    )

            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            z = (
                torch.tensor(self.gmm.sample(last_batch_samples_nbr)[0])
                .to(self.device)
                .type(torch.float)
            )
            x_gen = self.model.decoder(z)["reconstruction"].detach()

            if output_dir is not None:
                for j in range(last_batch_samples_nbr):
                    self.save_img(
                        x_gen[j],
                        output_dir,
                        "%08d.png" % int(batch_size * full_batch_nbr + j),
                    )

            x_gen_list.append(x_gen)

        if save_sampler_config:
            self.save(output_dir)

        if return_gen:
            return torch.cat(x_gen_list, dim=0)
