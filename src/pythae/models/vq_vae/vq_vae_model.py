import os
from typing import Optional

import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ...models import AE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_VAE_MLP
from .vq_vae_config import VQVAEConfig
from .vq_vae_utils import Quantizer, QuantizerEMA


class VQVAE(AE):
    r"""
    Vector Quantized-VAE model.

    Args:
        model_config (VQVAEConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: VQVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        AE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self._set_quantizer(model_config)

        self.model_name = "VQVAE"

    def _set_quantizer(self, model_config):

        if model_config.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of VQVAEConfig instance must be set to 'data_shape' where "
                "the shape of the data is (C, H, W ..). Unable to set quantizer"
            )

        x = torch.randn((2,) + self.model_config.input_dim)
        z = self.encoder(x).embedding
        if len(z.shape) == 2:
            z = z.reshape(
                z.shape[0], 1, int(z.shape[-1] ** 0.5), int(z.shape[-1] ** 0.5)
            )

        z = z.permute(0, 2, 3, 1)

        self.model_config.embedding_dim = z.shape[-1]
        if model_config.use_ema:
            self.quantizer = QuantizerEMA(model_config=model_config)

        else:
            self.quantizer = Quantizer(model_config=model_config)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        encoder_output = self.encoder(x)

        embeddings = encoder_output.embedding

        reshape_for_decoding = False

        if len(embeddings.shape) == 2:
            embeddings = embeddings.reshape(
                embeddings.shape[0],
                1,
                int(embeddings.shape[-1] ** 0.5),
                int(embeddings.shape[-1] ** 0.5),
            )

            reshape_for_decoding = True

        embeddings = embeddings.permute(0, 2, 3, 1)

        quantizer_output = self.quantizer(embeddings)

        quantized_embed = quantizer_output.quantized_vector
        quantized_indices = quantizer_output.quantized_indices

        if reshape_for_decoding:
            quantized_embed = quantized_embed.reshape(embeddings.shape[0], -1)

        recon_x = self.decoder(quantized_embed).reconstruction

        loss, recon_loss, vq_loss = self.loss_function(recon_x, x, quantizer_output)

        output = ModelOutput(
            recon_loss=recon_loss,
            vq_loss=vq_loss,
            loss=loss,
            recon_x=recon_x,
            z=quantized_embed,
            quantized_indices=quantized_indices,
        )

        return output

    def loss_function(self, recon_x, x, quantizer_output):

        recon_loss = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)

        vq_loss = quantizer_output.loss

        return (
            (recon_loss + vq_loss).mean(dim=0),
            recon_loss.mean(dim=0),
            vq_loss.mean(dim=0),
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = VQVAEConfig.from_json_file(path_to_model_config)

        return model_config
