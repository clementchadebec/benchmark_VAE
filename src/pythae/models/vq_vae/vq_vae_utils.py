import torch
import torch.nn as nn
from ..base.base_utils import ModelOuput

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
        
        # TODO should be reshaped before enterring here []

        distances = (z ** 2).sum(dim=-1, keepdim=True) \
                + (self.embeddings.weight ** 2).sum(dim=-1) \
                - 2 * z @ self.embeddings.weight.T
        
        closest = distances.argmin(-1).unsqueeze(-1)

        one_hot_encodings = nn.functionna
