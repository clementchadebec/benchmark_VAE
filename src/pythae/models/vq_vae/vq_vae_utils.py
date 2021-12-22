import torch
import torch.nn as nn
from ..base.base_utils import ModelOuput
import torch.nn.functional as F

from .vq_vae_config import VQVAEConfig

class Quantizer(nn.Module):

    def __init__(
        self,
        model_config: VQVAEConfig):

            self.model_config = model_config

            self.embedding_dim = model_config.embedding_dim
            self.num_embeddings = model_config.num_embeddings
            self.beta = model_config.beta

            self.embeddings = self.embedding = nn.Embedding(
                self.embedding_dim, self.num_embeddings)

            self.embeddings.weight.data.uniform_(
                -1 / self.K, 1 / self.K)

    def forward(self, z: torch.Tensor):

        z = z.permute(0, 2, 3, 1)        
        
        # TODO should be reshaped before enterring here []

        distances = (z ** 2).sum(dim=-1, keepdim=True) \
                + (self.embeddings.weight ** 2).sum(dim=-1) \
                - 2 * z @ self.embeddings.weight.T
        
        closest = distances.argmin(-1).unsqueeze(-1)

        one_hot_encoding = F.one_hot(
            cloesest, num_classes=self.num_embeddings
        )

        # quantization
        quantized = one_hot_encoding @ self.embeddings.weight
        quantized = quantized.reshape_as(z)

        commitment_loss = F.mse_loss(
            quantized.detach(),
            z,
            reduction='none'
        ).mean(dim=-1)

        embedding_loss =  F.mse_loss(
            quantized,
            z.detach(),
            reduction='none'
        ).mean(dim=-1)

        quantized = z + (quantized - z).detach()

        loss = commitment_loss * self.beta + embedding_loss

        output = ModelOutput(
            quantized_vector=quantized.permute(3, 1, 2),
            loss=loss,
            commitment_loss=commitment_loss,
            embedding_loss=embedding_loss,
        )