import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.base_utils import ModelOutput
from .vq_vae_config import VQVAEConfig

"""Code inspired from https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py"""


class Quantizer(nn.Module):
    def __init__(self, model_config: VQVAEConfig):

        nn.Module.__init__(self)

        self.model_config = model_config

        self.embedding_dim = model_config.embedding_dim
        self.num_embeddings = model_config.num_embeddings
        self.beta = model_config.beta

        self.embeddings = self.embedding = nn.Embedding(
            self.embedding_dim, self.num_embeddings
        )

        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, z: torch.Tensor):

        distances = (
            (z.reshape(-1, self.embedding_dim) ** 2).sum(dim=-1, keepdim=True)
            + (self.embeddings.weight.T ** 2).sum(dim=-1)
            - 2 * z.reshape(-1, self.embedding_dim) @ self.embeddings.weight
        )

        closest = distances.argmin(-1).unsqueeze(-1)

        one_hot_encoding = F.one_hot(closest, num_classes=self.num_embeddings).type(
            torch.float
        )

        # quantization
        quantized = one_hot_encoding @ self.embeddings.weight.T
        quantized = quantized.reshape_as(z)

        commitment_loss = F.mse_loss(
            quantized.detach().reshape(-1, self.embedding_dim),
            z.reshape(-1, self.embedding_dim),
            reduction="mean",
        )

        embedding_loss = F.mse_loss(
            quantized.reshape(-1, self.embedding_dim),
            z.detach().reshape(-1, self.embedding_dim),
            reduction="mean",
        ).mean(dim=-1)

        quantized = z + (quantized - z).detach()

        # loss = commitment_loss * self.beta + embedding_loss
        quantized = quantized.permute(0, 3, 1, 2)

        output = ModelOutput(
            quantized_vector=quantized,
            # loss=loss,
            commitment_loss=commitment_loss,
            embedding_loss=embedding_loss,
        )

        return output
