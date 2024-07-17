from typing import Optional, Any, Dict

import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..ae import AE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from .hrq_vae_config import HRQVAEConfig
from .hrq_vae_utils import HierarchicalResidualQuantizer


class HRQVAE(AE):
    r"""
    Hierarchical Residual Quantization-VAE model. Introduced in https://aclanthology.org/2022.acl-long.178/ (Hosking et al., ACL 2022)

    Args:
        model_config (HRQVAEConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    """

    def __init__(
        self,
        model_config: HRQVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        AE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self._set_quantizer(model_config)

        self.model_name = "HRQVAE"

    def _set_quantizer(self, model_config):
        if model_config.input_dim is None:
            raise AttributeError(
                "No input dimension provided !"
                "'input_dim' parameter of HRQVAEConfig instance must be set to 'data_shape' where "
                "the shape of the data is (C, H, W ..). Unable to set quantizer."
            )

        x = torch.randn((2,) + self.model_config.input_dim)
        z = self.encoder(x).embedding
        if len(z.shape) == 2:
            z = z.reshape(z.shape[0], 1, 1, -1)

        z = z.permute(0, 2, 3, 1)

        self.model_config.embedding_dim = z.shape[-1]

        self.quantizer = HierarchicalResidualQuantizer(model_config=model_config)

    def forward(self, inputs: Dict[str, Any], **kwargs) -> ModelOutput:
        """
        The VAE model

        Args:
            inputs (dict): A dict of samples

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]
        uses_ddp = kwargs.pop("uses_ddp", False)
        epoch = kwargs.pop("epoch", 0)

        encoder_output = self.encoder(x)

        embeddings = encoder_output.embedding

        reshape_for_decoding = False

        if len(embeddings.shape) == 2:
            embeddings = embeddings.reshape(embeddings.shape[0], 1, 1, -1)
            reshape_for_decoding = True

        embeddings = embeddings.permute(0, 2, 3, 1)

        quantizer_output = self.quantizer(embeddings, epoch=epoch, uses_ddp=uses_ddp)

        quantized_embed = quantizer_output.quantized_vector

        if reshape_for_decoding:
            quantized_embed = quantized_embed.reshape(embeddings.shape[0], -1)

        recon_x = self.decoder(quantized_embed).reconstruction

        loss, recon_loss, hrq_loss = self.loss_function(recon_x, x, quantizer_output)

        output = ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            hrq_loss=hrq_loss,
            recon_x=recon_x,
            z=quantized_embed,
            z_orig=quantizer_output.z_orig,
            quantized_indices=quantizer_output.quantized_indices,
            probs=quantizer_output.probs,
        )

        return output

    def loss_function(self, recon_x, x, quantizer_output):
        recon_loss = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)

        hrq_loss = quantizer_output.loss

        return (
            (recon_loss + hrq_loss).mean(dim=0),
            recon_loss.mean(dim=0),
            hrq_loss.mean(dim=0),
        )
