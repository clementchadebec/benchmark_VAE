from numpy import indices
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..base.base_utils import ModelOutput
from .sq_vae_config import SQVAEConfig

"""Code inspired from https://github.com/sony/sqvae/blob/main/vision/quantizer.py"""

def _sample_gumbel(shape, eps=1e-10, device='cuda'):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def _gumbel_softmax_sample(logits, temperature, device='cuda'):
    g = _sample_gumbel(logits.size(), device=device)
    y = logits + g
    return F.softmax(y / temperature, dim=-1)

class Quantizer(nn.Module):
    def __init__(self, model_config: SQVAEConfig):

        nn.Module.__init__(self)

        self.model_config = model_config

        self.embedding_dim = model_config.embedding_dim
        self.num_embeddings = model_config.num_embeddings

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, mu: torch.Tensor, var: torch.Tensor, temperature: torch.tensor, training_flag=True, det_quant_flag=False):

        precision_q = 1. / torch.clamp(var, min=1e-10)

        mu_flat = mu.reshape(-1, self.embedding_dim).unsqueeze(-1)
        
        weight = 0.5 * precision_q.permute(0, 2, 3, 1).contiguous().reshape(-1, self.embedding_dim).unsqueeze(-1)
        logit = -torch.sum(weight * ((mu_flat - self.embeddings.weight.T.unsqueeze(0)) ** 2), dim=1)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)

        if training_flag:
            encodings = _gumbel_softmax_sample(logit, temperature, device=mu.device)
            quantized = encodings @ self.embeddings.weight
            quantized = quantized.reshape_as(mu)
            indices = torch.argmax(logit, dim=1).unsqueeze(1)
            #avg_probs = torch.mean(probabilities.detach(), dim=0)

        else:
            if det_quant_flag:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                one_hot_encoding = (
                    F.one_hot(indices, num_classes=self.num_embeddings)
                    .type(torch.float)
                    .squeeze(1)
                )
                #avg_probs = torch.mean(one_hot_encoding, dim=0)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(mu.shape[0], mu.shape[2], mu.shape[1])
                one_hot_encoding = (
                    F.one_hot(indices, num_classes=self.num_embeddings)
                    .type(torch.float)
                    .squeeze(1)
                )
                #avg_probs = torch.mean(probabilities, dim=0)
            quantized = one_hot_encoding @ self.embeddings.weight
        quantized = quantized.reshape_as(mu)
        quantized_indices = indices.reshape(mu.shape[0], mu.shape[1], mu.shape[2])

        # Latent loss
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / mu.shape[0]
        kld_continous = ((quantized.reshape(-1, self.embedding_dim) - mu.reshape(-1, self.embedding_dim)) ** 2 * weight.reshape(-1, self.embedding_dim)).sum(dim=-1).mean(dim=0)
        loss = kld_discrete + kld_continous

        quantized = quantized.permute(0, 3, 1, 2)

        output = ModelOutput(
            quantized_vector=quantized,
            quantized_indices=quantized_indices.unsqueeze(1),
            loss=loss,
        )

        return output
