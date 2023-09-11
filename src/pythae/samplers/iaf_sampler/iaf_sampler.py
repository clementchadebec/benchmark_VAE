import os
import shutil
from typing import Union

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, Dataset

from ...data.datasets import collate_dataset_output
from ...data.preprocessors import DataProcessor
from ...models import BaseAE
from ...models.normalizing_flows import IAF, IAFConfig, NFModel
from ...trainers import BaseTrainer, BaseTrainerConfig
from ..base.base_sampler import BaseSampler
from .iaf_sampler_config import IAFSamplerConfig


class IAFSampler(BaseSampler):
    """Fits an Inverse Autoregressive Flow in the Autoencoder's latent space.

    Args:
        model (BaseAE): The AE model to sample from
        sampler_config (IAFSamplerConfig): A IAFSamplerConfig instance containing
            the main parameters of the sampler. If None, a pre-defined configuration is used.
            Default: None

    .. note::

        The method :class:`~pythae.samplers.IAFSampler.fit` must be called to fit the
        sampler before sampling.
    """

    def __init__(self, model: BaseAE, sampler_config: IAFSamplerConfig = None):

        self.is_fitted = False

        if sampler_config is None:
            sampler_config = IAFSamplerConfig()

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

        self.prior = MultivariateNormal(
            torch.zeros(model.model_config.latent_dim).to(self.device),
            torch.eye(model.model_config.latent_dim).to(self.device),
        )

        iaf_config = IAFConfig(
            input_dim=(model.model_config.latent_dim,),
            n_made_blocks=sampler_config.n_made_blocks,
            n_hidden_in_made=sampler_config.n_hidden_in_made,
            hidden_size=sampler_config.hidden_size,
            include_batch_norm=sampler_config.include_batch_norm,
        )

        iaf_model = IAF(model_config=iaf_config)

        self.flow_contained_model = NFModel(self.prior, iaf_model)

        self.flow_contained_model.to(self.device)

    def fit(
        self,
        train_data: Union[torch.Tensor, np.ndarray, Dataset],
        eval_data: Union[torch.Tensor, np.ndarray, Dataset, None] = None,
        training_config: BaseTrainerConfig = None,
        batch_size: int = 64,
    ):
        """Method to fit the sampler from the training data

        Args:
            train_data (Union[torch.Tensor, np.ndarray, Dataset]): The train data needed to
                retrieve the training embeddings and fit the flows in the latent space.
            eval_data (Union[torch.Tensor, np.ndarray, Dataset]): The train data needed to retrieve
                the evaluation embeddings and fit the flows in the latent space.
            training_config (BaseTrainerConfig): the training config to use to fit the flow.
            batch_size (int): The batch size to use to retrieve the embeddings. Default: 64.
        """
        data_processor = DataProcessor()
        if not isinstance(train_data, Dataset):
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
                    encoder_output = self.model(inputs)
                    z_ = encoder_output.z
                    z.append(z_)

        except RuntimeError:
            for _, inputs in enumerate(train_loader):
                inputs = self._set_inputs_to_device(inputs)
                encoder_output = self.model(inputs)
                z_ = encoder_output.z.detach()
                z.append(z_)

        train_data = torch.cat(z)
        train_dataset = data_processor.to_dataset(train_data)

        eval_dataset = None

        if eval_data is not None:

            if not isinstance(eval_data, Dataset):
                eval_data = data_processor.process_data(eval_data)
                eval_dataset = data_processor.to_dataset(eval_data)

            else:
                eval_dataset = eval_data

            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_dataset_output,
            )

            z = []

            try:
                with torch.no_grad():
                    for _, inputs in enumerate(eval_loader):
                        inputs = self._set_inputs_to_device(inputs)
                        encoder_output = self.model(inputs)
                        z_ = encoder_output.z
                        z.append(z_)

            except RuntimeError:
                for _, inputs in enumerate(eval_loader):
                    inputs = self._set_inputs_to_device(inputs)
                    encoder_output = self.model(inputs)
                    z_ = encoder_output.z.detach()
                    z.append(z_)

            eval_data = torch.cat(z)
            eval_dataset = data_processor.to_dataset(eval_data)

        trainer = BaseTrainer(
            model=self.flow_contained_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
        )

        trainer.train()

        self.iaf_model = IAF.load_from_folder(
            os.path.join(trainer.training_dir, "final_model")
        ).to(self.device)

        shutil.rmtree(trainer.training_dir)

        self.is_fitted = True

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
                "The sampler needs to be fitted by calling sampler.fit() method"
                "before sampling."
            )

        full_batch_nbr = int(num_samples / batch_size)
        last_batch_samples_nbr = num_samples % batch_size

        x_gen_list = []

        for i in range(full_batch_nbr):

            u = self.prior.sample((batch_size,))
            z = self.iaf_model.inverse(u).out
            x_gen = self.model.decoder(z).reconstruction.detach()

            if output_dir is not None:
                for j in range(batch_size):
                    self.save_img(
                        x_gen[j], output_dir, "%08d.png" % int(batch_size * i + j)
                    )

            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            u = self.prior.sample((last_batch_samples_nbr,))
            z = self.iaf_model.inverse(u).out
            x_gen = self.model.decoder(z).reconstruction.detach()

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
