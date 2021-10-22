import datetime
import logging
import os

import torch
import torch.nn as nn
import shutil

from ...data.preprocessors import DataProcessor
from torch.utils.data import DataLoader
from ..base.base_sampler import BaseSampler
from ...models.base.base_utils import ModelOuput
from ...models.nn import BaseEncoder, BaseDecoder
from ...models import RHVAE, RHVAEConfig
from ...models import VAE, VAEConfig
from .two_stage_sampler_config import TwoStageVAESamplerConfig
from ...trainers import BaseTrainingConfig
from ...pipelines import TrainingPipeline

class SecondEncoder(BaseEncoder):

            def __init__(self, model: VAE, sampler_config: TwoStageVAESamplerConfig):

                BaseEncoder.__init__(self)
            
                layers = []

                layers.append(
                    nn.Sequential(
                        nn.Linear(model.latent_dim, sampler_config.second_layers_dim),
                        nn.ReLU()
                    )
                )

                for i in range(sampler_config.second_stage_depth-1):
                    layers.append(nn.Sequential(
                        nn.Linear(
                            sampler_config.second_layers_dim, sampler_config.second_layers_dim),
                        nn.ReLU()
                    )
                    )

                self.layers = nn.Sequential(*layers)
                self.mu = nn.Linear(sampler_config.second_layers_dim, model.latent_dim)
                self.std = nn.Linear(sampler_config.second_layers_dim, model.latent_dim)

            def forward(self, z:torch.Tensor):
                out = self.layers(z)

                output = ModelOuput(
                    embedding=self.mu(out),
                    log_covariance=self.std(out)
                )

                return output

class SecondDecoder(BaseDecoder):

            def __init__(self, model: VAE, sampler_config: TwoStageVAESamplerConfig):

                BaseDecoder.__init__(self)

                self.gamma_z = nn.Parameter(torch.ones(1, 1), requires_grad=True)
            
                layers = []

                layers.append(
                    nn.Sequential(
                        nn.Linear(model.latent_dim, sampler_config.second_layers_dim),
                        nn.ReLU()
                    )
                )

                for i in range(sampler_config.second_stage_depth-1):
                    layers.append(nn.Sequential(
                        nn.Linear(
                            sampler_config.second_layers_dim, sampler_config.second_layers_dim),
                        nn.ReLU())
                    )

                self.layers = nn.Sequential(*layers)
                self.reconstruction = nn.Linear(sampler_config.second_layers_dim, model.latent_dim)
                

            def forward(self, u:torch.Tensor):
                out = self.layers(u)

                z = self.reconstruction(out)

                output = ModelOuput(
                    reconstruction= z + self.gamma_z * torch.randn_like(z)
                )

                return output



class TwoStageVAESampler(BaseSampler):
    """Two Stage VAE sampler class.
    This is an implementation of the two stage VAE sampler
    (https://openreview.net/pdf?id=B1e0X3C9tQ)

    Args:
        model (VAE): The VAE model to sample from
        sampler_config (TwoStageVAESamplerConfig): A TwoStageVAESamplerConfig instance containing 
            the main parameters of the sampler. If None, a pre-defined configuration is used. 
            Default: None
    """

    def __init__(self, model: VAE, sampler_config: TwoStageVAESamplerConfig = None):

        assert issubclass(model.__class__, VAE), ("The TwoStageVAESampler is only" 
            f"applicable for VAE based models. Got {model.__class__}.")

        self.is_fitted = False

        if sampler_config is None:
            sampler_config = TwoStageVAESamplerConfig()

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

        second_vae_config = VAEConfig(
            latent_dim=model.model_config.latent_dim,
            reconstruction_loss=sampler_config.reconstruction_loss
        )

        self.second_vae = VAE(
            model_config=second_vae_config,
            encoder=SecondEncoder(model, sampler_config),
            decoder=SecondDecoder(model, sampler_config))

        self.second_vae.model_name = "Second_Stage_VAE"

        self.second_vae.to(self.device)


    def fit(self, train_data, eval_data=None, training_config: BaseTrainingConfig=None):
        """Method to fit the sampler from the training data

        Args:
            train_data (torch.Tensor): The train data needed to retreive the training embeddings
                    and fit the mixture in the latent space. Must be of shape n_imgs x im_channels x 
                    ... and in range [0-1]
            train_data (torch.Tensor): The train data needed to retreive the training embeddings
                    and fit the mixture in the latent space. Must be of shape n_imgs x im_channels x 
                    ... and in range [0-1]
            path_to_training_config (BaseTrainingConfig): path to the training config to use to fit 
                the second VAE. 
        """

        assert (
            train_data.max() >= 1 and train_data.min() >= 0
        ), "Train data must in the range [0-1]"

        data_processor = DataProcessor()
        train_data = data_processor.process_data(train_data.to(self.device))
        train_dataset = data_processor.to_dataset(train_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=False)

        z = []

        with torch.no_grad():
            for _, inputs in enumerate(train_loader):
                encoder_output = self.model.encoder(inputs["data"].to(self.device))
                mean_z, log_std_z = encoder_output.embedding, encoder_output.log_covariance
                z_data = mean_z + torch.randn_like(log_std_z) * log_std_z.exp()
                z.append(z_data)

        train_data = torch.cat(z)

        pipeline = TrainingPipeline(training_config=training_config, model=self.second_vae)
        pipeline(train_data=train_data, eval_data=eval_data)
        shutil.rmtree(pipeline.trainer.training_dir)


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
                "The sampler needs to be fitted by calling smapler.fit() method"
                "before sampling."
            )

        full_batch_nbr = int(num_samples / batch_size)
        last_batch_samples_nbr = num_samples % batch_size

        x_gen_list = []

        for i in range(full_batch_nbr):

            u = torch.randn(batch_size, self.model.latent_dim).to(self.device)
            z = self.second_vae.decoder(u).reconstruction
            z = z + self.second_vae.decoder.gamma_z \
                * torch.randn(batch_size, self.model.latent_dim).to(self.device)
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
            z = z + self.second_vae.decoder.gamma_z \
                * torch.randn(last_batch_samples_nbr, self.model.latent_dim).to(self.device)
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
