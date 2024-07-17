import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from ..base.base_utils import ModelOutput
from .hrq_vae_config import HRQVAEConfig


class HierarchicalResidualQuantizer(nn.Module):
    def __init__(self, model_config: HRQVAEConfig):
        nn.Module.__init__(self)

        self.model_config = model_config

        self.embedding_dim = model_config.embedding_dim
        self.num_embeddings = model_config.num_embeddings
        self.num_levels = model_config.num_levels

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(self.num_embeddings, self.embedding_dim)
                for hix in range(self.num_levels)
            ]
        )

        init_scale = model_config.init_scale
        init_decay_weight = model_config.init_decay_weight
        for hix, embedding in enumerate(self.embeddings):
            scale = init_scale * init_decay_weight**hix / sqrt(self.embedding_dim)
            embedding.weight.data.uniform_(-1.0 * scale, scale)

            # Normalise onto sphere
            embedding.weight.data = (
                embedding.weight.data
                / torch.linalg.vector_norm(embedding.weight, dim=1, keepdim=True)
                * init_scale
                * init_decay_weight**hix
            )

    def forward(self, z: torch.Tensor, epoch: int, uses_ddp: bool = False):
        if uses_ddp:
            raise Exception("HRQVAE doesn't currently support DDP :(")

        input_shape = z.shape

        z = z.reshape(-1, self.embedding_dim)

        loss = torch.zeros(z.shape[0]).to(z.device)

        resid_error = z

        quantized = []
        codes = []
        all_probs = []

        for head_ix, embedding in enumerate(self.embeddings):
            if head_ix > 0:
                resid_error = z - torch.sum(torch.cat(quantized, dim=1), dim=1)

            distances = -1.0 * (
                torch.sum(resid_error**2, dim=-1, keepdim=True)
                + torch.sum(embedding.weight**2, dim=-1)
                - 2 * torch.matmul(resid_error, embedding.weight.T)
            )

            gumbel_sched_weight = torch.exp(
                -torch.tensor(float(epoch))
                / float(self.model_config.temp_schedule_gamma * 1.5**head_ix)
            )
            gumbel_temp = max(gumbel_sched_weight, 0.5)

            if self.training:

                sample_onehot = F.gumbel_softmax(
                    distances, tau=gumbel_temp, hard=True, dim=-1
                )
            else:
                indices = torch.argmax(distances, dim=-1)
                sample_onehot = F.one_hot(
                    indices, num_classes=self.num_embeddings
                ).float()

            probs = F.softmax(distances / gumbel_temp, dim=-1)

            # KL loss
            prior = (
                torch.ones_like(distances).detach()
                / torch.ones_like(distances).sum(-1, keepdim=True).detach()
            )
            kl_loss = torch.nn.KLDivLoss(reduction="none")
            kl = kl_loss(nn.functional.log_softmax(distances, dim=-1), prior).sum(
                dim=-1
            )
            loss += kl * self.model_config.kl_weight

            # quantization
            this_quantized = sample_onehot @ embedding.weight
            this_quantized = this_quantized.reshape_as(z)

            quantized.append(this_quantized.unsqueeze(-2))
            codes.append(torch.argmax(sample_onehot, dim=-1).unsqueeze(-1))
            all_probs.append(probs.unsqueeze(-2))

        quantized = torch.cat(quantized, dim=-2)
        quantized_indices = torch.cat(codes, dim=-1)
        all_probs = torch.cat(all_probs, dim=-2)

        # Calculate the norm loss
        if self.model_config.norm_loss_weight is not None:
            upper_norms = torch.linalg.vector_norm(quantized[:, :-1, :], dim=-1)
            lower_norms = torch.linalg.vector_norm(quantized[:, 1:, :], dim=-1)
            norm_loss = (
                torch.max(
                    lower_norms / upper_norms * self.model_config.norm_loss_scale,
                    torch.ones_like(lower_norms),
                )
                - 1.0
            ) ** 2

            loss += norm_loss.mean(dim=1) * self.model_config.norm_loss_weight

        # Depth drop out
        if self.training:

            drop_dist = torch.distributions.Bernoulli(
                1 - self.model_config.depth_drop_rate
            )

            mask = drop_dist.sample(sample_shape=(*quantized.shape[:-1], 1))

            mask = torch.cumprod(mask, dim=1).to(quantized.device)
            quantized = quantized * mask

        quantized = quantized.sum(dim=-2).reshape(*input_shape)
        quantized = quantized.permute(0, 3, 1, 2)

        loss = loss.reshape(input_shape[0], -1).mean(dim=1)

        quantized_indices = quantized_indices.reshape(
            *input_shape[:-1], self.num_levels
        )

        output = ModelOutput(
            z_orig=z,
            quantized_vector=quantized,
            quantized_indices=quantized_indices,
            loss=loss,
            probs=all_probs,
        )

        return output
