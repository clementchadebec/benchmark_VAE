import os
import shutil
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ...data.datasets import collate_dataset_output
from ...data.preprocessors import DataProcessor
from ...models import VQVAE
from ...models.normalizing_flows import PixelCNN, PixelCNNConfig
from ...trainers import BaseTrainer, BaseTrainerConfig
from ..base.base_sampler import BaseSampler
from .pixelcnn_sampler_config import PixelCNNSamplerConfig


class PixelCNNSampler(BaseSampler):
    """Fits a PixelCNN in the VQVAE's latent space.

    Args:
        model (VQVAE): The AE model to sample from
        sampler_config (PixelCNNSamplerConfig): A PixelCNNSamplerConfig instance containing
            the main parameters of the sampler. If None, a pre-defined configuration is used.
            Default: None

    .. note::

        The method :class:`~pythae.samplers.MAFSampler.fit` must be called to fit the
        sampler before sampling.
    """

    def __init__(self, model: VQVAE, sampler_config: PixelCNNSamplerConfig = None):

        self.is_fitted = False

        if sampler_config is None:
            sampler_config = PixelCNNSamplerConfig()

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

        # get size of codes
        x_dumb = torch.randn((2,) + model.model_config.input_dim).to(self.device)
        out_dumb = model({"data": x_dumb})
        quant_dumb = out_dumb.quantized_indices
        z_dumb = out_dumb.z

        self.needs_reshape = False
        if len(z_dumb.shape) == 2:
            self.needs_reshape = True

        pixelcnn_config = PixelCNNConfig(
            input_dim=quant_dumb.shape[1:],
            n_embeddings=model.model_config.num_embeddings,
            n_layers=sampler_config.n_layers,
            kernel_size=sampler_config.kernel_size,
        )

        self.pixelcnn_model = PixelCNN(model_config=pixelcnn_config).to(self.device)

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
                retrieve the training embeddings and fit the PixelCNN model in the latent space.
            eval_data (Union[torch.Tensor, np.ndarray, Dataset]): The train data needed to retrieve
                the evaluation embeddings and fit the PixelCNN model in the latent space.
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

        with torch.no_grad():
            for _, inputs in enumerate(train_loader):
                inputs = self._set_inputs_to_device(inputs)
                model_output = self.model(inputs)
                mean_z = model_output.quantized_indices
                z.append(
                    mean_z.reshape(
                        (mean_z.shape[0],) + self.pixelcnn_model.model_config.input_dim
                    )
                )

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

            with torch.no_grad():
                for _, inputs in enumerate(eval_loader):
                    inputs = self._set_inputs_to_device(inputs)
                    model_output = self.model(inputs)
                    mean_z = model_output.quantized_indices
                    z.append(
                        mean_z.reshape(
                            (mean_z.shape[0],)
                            + self.pixelcnn_model.model_config.input_dim
                        )
                    )

            eval_data = torch.cat(z)
            eval_dataset = data_processor.to_dataset(eval_data)

        trainer = BaseTrainer(
            model=self.pixelcnn_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=training_config,
        )

        trainer.train()

        self.pixelcnn_model = PixelCNN.load_from_folder(
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

            z = torch.zeros(
                (batch_size,) + self.pixelcnn_model.model_config.input_dim
            ).to(self.device)
            for k in range(self.pixelcnn_model.model_config.input_dim[-2]):
                for l in range(self.pixelcnn_model.model_config.input_dim[-1]):
                    out = self.pixelcnn_model({"data": z}).out.squeeze(2)
                    probs = F.softmax(out[:, :, k, l], dim=-1)
                    z[:, :, k, l] = torch.multinomial(probs, 1).float()

            z_quant = self.model.quantizer.embeddings(z.reshape(z.shape[0], -1).long())

            if self.needs_reshape:
                z_quant = z_quant.reshape(z.shape[0], -1)
            else:
                z_quant = z_quant.reshape(
                    z.shape[0], z.shape[2], z.shape[3], -1
                ).permute(0, 3, 1, 2)

            x_gen = self.model.decoder(z_quant).reconstruction.detach()

            if output_dir is not None:
                for j in range(batch_size):
                    self.save_img(
                        x_gen[j], output_dir, "%08d.png" % int(batch_size * i + j)
                    )

            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            z = torch.zeros(
                (last_batch_samples_nbr,) + self.pixelcnn_model.model_config.input_dim
            ).to(self.device)
            for k in range(self.pixelcnn_model.model_config.input_dim[-2]):
                for l in range(self.pixelcnn_model.model_config.input_dim[-1]):
                    out = self.pixelcnn_model({"data": z}).out.squeeze(2)
                    probs = F.softmax(out[:, :, k, l], dim=-1)
                    z[:, :, k, l] = torch.multinomial(probs, 1).float()

            z_quant = self.model.quantizer.embeddings(z.reshape(z.shape[0], -1).long())

            if self.needs_reshape:
                z_quant = z_quant.reshape(z_quant.shape[0], -1)
            else:
                z_quant = z_quant.reshape(
                    z.shape[0], z.shape[2], z.shape[3], -1
                ).permute(0, 3, 1, 2)

            x_gen = self.model.decoder(z_quant).reconstruction.detach()

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
