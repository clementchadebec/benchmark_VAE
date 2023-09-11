import os
import shutil
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ...data.datasets import collate_dataset_output
from ...data.preprocessors import DataProcessor
from ...models import VAE, VAEConfig
from ...models.base.base_utils import ModelOutput
from ...models.nn import BaseDecoder, BaseEncoder
from ...trainers import BaseTrainer, BaseTrainerConfig
from ..base import BaseSampler
from .two_stage_sampler_config import TwoStageVAESamplerConfig


class SecondEncoder(BaseEncoder):
    def __init__(self, model: VAE, sampler_config: TwoStageVAESamplerConfig):

        BaseEncoder.__init__(self)

        layers = []

        layers.append(
            nn.Sequential(
                nn.Linear(model.latent_dim, sampler_config.second_layers_dim), nn.ReLU()
            )
        )

        for i in range(sampler_config.second_stage_depth - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(
                        sampler_config.second_layers_dim,
                        sampler_config.second_layers_dim,
                    ),
                    nn.ReLU(),
                )
            )

        self.layers = nn.Sequential(*layers)
        self.mu = nn.Linear(sampler_config.second_layers_dim, model.latent_dim)
        self.std = nn.Linear(sampler_config.second_layers_dim, model.latent_dim)

    def forward(self, z: torch.Tensor):
        out = self.layers(z)

        output = ModelOutput(embedding=self.mu(out), log_covariance=self.std(out))

        return output


class SecondDecoder(BaseDecoder):
    def __init__(self, model: VAE, sampler_config: TwoStageVAESamplerConfig):

        BaseDecoder.__init__(self)

        self.gamma_z = nn.Parameter(torch.ones(1, 1), requires_grad=True)

        layers = []

        layers.append(
            nn.Sequential(
                nn.Linear(model.latent_dim, sampler_config.second_layers_dim), nn.ReLU()
            )
        )

        for i in range(sampler_config.second_stage_depth - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(
                        sampler_config.second_layers_dim,
                        sampler_config.second_layers_dim,
                    ),
                    nn.ReLU(),
                )
            )

        self.layers = nn.Sequential(*layers)
        self.reconstruction = nn.Linear(
            sampler_config.second_layers_dim, model.latent_dim
        )

    def forward(self, u: torch.Tensor):
        out = self.layers(u)

        z = self.reconstruction(out)

        output = ModelOutput(reconstruction=z + self.gamma_z * torch.randn_like(z))

        return output


class TwoStageVAESampler(BaseSampler):
    """Fits a second VAE in the Autoencoder's latent space.

    Args:
        model (VAE): The VAE model to sample from
        sampler_config (TwoStageVAESamplerConfig): A TwoStageVAESamplerConfig instance containing
            the main parameters of the sampler. If None, a pre-defined configuration is used.
            Default: None

    .. note::

        The method :class:`~pythae.samplers.TwoStageVAESampler.fit` must be called to fit the
        sampler before sampling.
    """

    def __init__(self, model: VAE, sampler_config: TwoStageVAESamplerConfig = None):

        assert issubclass(model.__class__, VAE), (
            "The TwoStageVAESampler is only"
            f"applicable for VAE based models. Got {model.__class__}."
        )

        self.is_fitted = False

        if sampler_config is None:
            sampler_config = TwoStageVAESamplerConfig()

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

        second_vae_config = VAEConfig(
            latent_dim=model.model_config.latent_dim,
            reconstruction_loss=sampler_config.reconstruction_loss,
        )

        self.second_vae = VAE(
            model_config=second_vae_config,
            encoder=SecondEncoder(model, sampler_config),
            decoder=SecondDecoder(model, sampler_config),
        )

        self.second_vae.model_name = "Second_Stage_VAE"

        self.second_vae.to(self.device)

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
                retrieve the training embeddings and fit the second VAE in the latent space.
            eval_data (Union[torch.Tensor, np.ndarray, Dataset]): The train data needed to retrieve
                the evaluation embeddings and fit the second VAE in the latent space.
            training_config (BaseTrainerConfig): the training config to use to fit the second VAE.
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
            model=self.second_vae,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
        )

        trainer.train()

        self.second_vae = VAE.load_from_folder(
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

            u = torch.randn(batch_size, self.model.latent_dim).to(self.device)
            z = self.second_vae.decoder(u).reconstruction
            x_gen = self.model.decoder(z)["reconstruction"].detach()

            if output_dir is not None:
                for j in range(batch_size):
                    self.save_img(
                        x_gen[j], output_dir, "%08d.png" % int(batch_size * i + j)
                    )

            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            u = torch.randn(last_batch_samples_nbr, self.model.latent_dim).to(
                self.device
            )
            z = self.second_vae.decoder(u).reconstruction
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
