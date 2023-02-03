from dis import dis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ..base.base_utils import ModelOutput
from .vq_vae_config import VQVAEConfig

"""Code inspired from https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py"""


class Quantizer(nn.Module):
    def __init__(self, model_config: VQVAEConfig):

        nn.Module.__init__(self)

        self.model_config = model_config

        self.embedding_dim = model_config.embedding_dim
        self.num_embeddings = model_config.num_embeddings
        self.commitment_loss_factor = model_config.commitment_loss_factor
        self.quantization_loss_factor = model_config.quantization_loss_factor

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, z: torch.Tensor, uses_ddp=False):

        distances = torch.cdist(z.reshape(-1, self.embedding_dim), self.embeddings.weight, p=2)

        closest = distances.argmin(-1).unsqueeze(-1)

        quantized_indices = closest.reshape(z.shape[0], z.shape[1], z.shape[2])

        one_hot_encoding = (
            F.one_hot(closest, num_classes=self.num_embeddings)
            .type(torch.float)
            .squeeze(1)
        )

        # quantization
        quantized = one_hot_encoding @ self.embeddings.weight
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

        loss = (
            commitment_loss * self.commitment_loss_factor
            + embedding_loss * self.quantization_loss_factor
        )
        quantized = quantized.permute(0, 3, 1, 2)

        output = ModelOutput(
            quantized_vector=quantized,
            quantized_indices=quantized_indices.unsqueeze(1),
            loss=loss,
        )

        return output


class QuantizerEMA(nn.Module):
    def __init__(self, model_config: VQVAEConfig):

        nn.Module.__init__(self)

        self.model_config = model_config

        self.embedding_dim = model_config.embedding_dim
        self.num_embeddings = model_config.num_embeddings
        self.commitment_loss_factor = model_config.commitment_loss_factor
        self.decay = model_config.decay

        embedding = torch.empty(self.num_embeddings, self.embedding_dim)
        nn.init.kaiming_uniform_(embedding)

        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer('ema_embed', embedding.clone())
        self.register_buffer("embeddings", embedding)

    def forward(self, z: torch.Tensor, uses_ddp: bool=False):

        distances = torch.cdist(z.reshape(-1, self.embedding_dim), self.embeddings, p=2)

        closest = distances.argmin(-1).unsqueeze(-1)

        quantized_indices = closest.reshape(z.shape[0], z.shape[1], z.shape[2])

        one_hot_encoding = (
            F.one_hot(closest, num_classes=self.num_embeddings)
            .type(torch.float)
            .squeeze(1)
        )

        # quantization
        quantized = one_hot_encoding @ self.embeddings
        quantized = quantized.reshape_as(z)

        if self.training:

            n_i = torch.sum(one_hot_encoding, dim=0)

            print("before first EMA")

            if uses_ddp:
                print("IN first all reduce")
                dist.all_reduce(n_i)

            print("OUT first all reduce")



            # ema update
            #self.cluster_size.mul_(self.decay).add_(n_i, alpha = (1 - self.decay))

            self.cluster_size = self.cluster_size * self.decay + n_i * (1 - self.decay)

            dw = one_hot_encoding.T @ z.reshape(-1, self.embedding_dim)

            if uses_ddp:
                dist.all_reduce(dw)

            #self.ema_embed.mul_(self.decay).add_(dw, alpha=(1 - self.decay))

            self.ema_embed = nn.Parameter(
                self.ema_embed * self.decay + dw * (1 - self.decay)
            )

            n = torch.sum(self.cluster_size)

            cluster_size = (
                (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5)
            ) * n

            ema_embedding_normalized = self.ema_embed / cluster_size.unsqueeze(-1)

            self.ema_embed.data.copy_(ema_embedding_normalized)

            #self.embeddings.weight = nn.Parameter(
            #    self.ema_embed / self.cluster_size.unsqueeze(-1)
            #)

        commitment_loss = F.mse_loss(
            quantized.detach().reshape(-1, self.embedding_dim),
            z.reshape(-1, self.embedding_dim),
            reduction="mean",
        )

        quantized = z + (quantized - z).detach()

        loss = commitment_loss * self.commitment_loss_factor
        quantized = quantized.permute(0, 3, 1, 2)

        output = ModelOutput(
            quantized_vector=quantized,
            quantized_indices=quantized_indices.unsqueeze(1),
            loss=loss,
        )

        return output
