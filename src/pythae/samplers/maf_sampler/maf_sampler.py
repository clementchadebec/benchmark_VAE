import datetime
import logging
import os
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal

from ...data.preprocessors import DataProcessor
from ...models import BaseAE, AE, AEConfig
from ...models.normalizing_flows import MAF, MAFConfig, NFModel
from ...models.base.base_utils import ModelOutput
from ...models.nn import BaseDecoder, BaseEncoder
from ...pipelines import TrainingPipeline
from ...trainers import BaseTrainerConfig
from ..base.base_sampler import BaseSampler
from .maf_sampler_config import MAFSamplerConfig


class MAFSampler(BaseSampler):
    """MAF sampler.

    Args:
        model (BaseAE): The AE model to sample from
        sampler_config (MAFSamplerConfig): A MAFSamplerConfig instance containing
            the main parameters of the sampler. If None, a pre-defined configuration is used.
            Default: None

    .. note::

        The method :class:`~pythae.samplers.MAFSampler.fit` must be called to fit the
        sampler before sampling.
    """

    def __init__(self, model: BaseAE, sampler_config: MAFSamplerConfig = None):

        self.is_fitted = False

        if sampler_config is None:
            sampler_config = MAFSamplerConfig()

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

        self.prior = MultivariateNormal(
            torch.zeros(model.model_config.latent_dim).to(self.device),
            torch.eye(model.model_config.latent_dim).cuda(self.device)
        )

        maf_config = MAFConfig(
            input_dim=(model.model_config.latent_dim,),
            n_made_blocks=sampler_config.n_made_blocks,
            n_hidden_in_made=sampler_config.n_hidden_in_made,
            hidden_size=sampler_config.hidden_size,
            include_batch_norm=sampler_config.include_batch_norm
        )

        maf_model = MAF(
            model_config=maf_config
        )

        self.flow_contained_model = NFModel(self.prior, maf_model)

        self.flow_contained_model.to(self.device)

    def fit(
        self, train_data, eval_data=None, training_config: BaseTrainerConfig = None
    ):
        """Method to fit the sampler from the training data

        Args:
            train_data (torch.Tensor): The train data needed to retreive the training embeddings
                    and fit the mixture in the latent space. Must be of shape n_imgs x im_channels x
                    ... and in range [0-1]
            eval_data (torch.Tensor): The train data needed to retreive the evaluation embeddings
                    and fit the mixture in the latent space. Must be of shape n_imgs x im_channels x
                    ... and in range [0-1]
            training_config (BaseTrainerConfig): the training config to use to fit the flow.
        """

        assert (
            train_data.max() >= 1 and train_data.min() >= 0
        ), "Train data must in the range [0-1]"

        data_processor = DataProcessor()
        train_data = data_processor.process_data(train_data)
        train_dataset = data_processor.to_dataset(train_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

        z = []

        with torch.no_grad():
            for _, inputs in enumerate(train_loader):
                encoder_output = self.model.encoder(inputs["data"].to(self.device))
                mean_z = encoder_output.embedding
                z.append(mean_z)

        train_data = torch.cat(z)

        if eval_data is not None:

            assert (
                eval_data.max() >= 1 and eval_data.min() >= 0
            ), "Eval data must in the range [0-1]"

            eval_data = data_processor.process_data(eval_data.to(self.device))
            eval_dataset = data_processor.to_dataset(eval_data)
            eval_loader = DataLoader(
                dataset=eval_dataset, batch_size=100, shuffle=False
            )

            z = []

            with torch.no_grad():
                for _, inputs in enumerate(eval_loader):
                    encoder_output = self.model.encoder(inputs["data"].to(self.device))
                    mean_z = encoder_output.embedding
                    z.append(mean_z)

            eval_data = torch.cat(z)

        pipeline = TrainingPipeline(
            training_config=training_config, model=self.flow_contained_model
        )

        pipeline(train_data=train_data, eval_data=eval_data)

        self.maf_model = MAF.load_from_folder(
            os.path.join(pipeline.trainer.training_dir, "final_model")
        ).to(self.device)

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
                "The sampler needs to be fitted by calling sampler.fit() method"
                "before sampling."
            )

        full_batch_nbr = int(num_samples / batch_size)
        last_batch_samples_nbr = num_samples % batch_size

        x_gen_list = []

        for i in range(full_batch_nbr):

            u = self.prior.sample((batch_size,))
            z = self.maf_model.inverse(u).out
            x_gen = self.model.decoder(z).reconstruction.detach()

            if output_dir is not None:
                for j in range(batch_size):
                    self.save_img(
                        x_gen[j], output_dir, "%08d.png" % int(batch_size * i + j)
                    )

            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            u = self.prior.sample((last_batch_samples_nbr,))
            z = self.maf_model.inverse(u).out
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
