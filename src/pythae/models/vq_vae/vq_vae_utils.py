import torch
import torch.nn as nn

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

            self.embedding.weight.data.uniform_(
                -1 / self.K, 1 / self.K)