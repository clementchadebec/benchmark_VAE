from typing import Optional

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_SVAE_MLP
from ..vae import VAE
from .svae_config import SVAEConfig
from .svae_utils import ive


class SVAE(VAE):
    r"""
    :math:`\mathcal{S}`-VAE model.

    Args:
        model_config (SVAEConfig): The Variational Autoencoder configuration setting the main
        parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: SVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "SVAE"

        if encoder is None:
            encoder = Encoder_SVAE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

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

        loc, log_concentration = (
            encoder_output.embedding,
            encoder_output.log_concentration,
        )

        # normalize mean
        loc = loc / loc.norm(dim=-1, keepdim=True)

        concentration = torch.nn.functional.softplus(log_concentration) + 1
        z = self._sample_von_mises(loc, concentration)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, loc, concentration, z)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def loss_function(self, recon_x, x, loc, concentration, z):

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = (
                0.5
                * F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1)
            )

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = self._compute_kl(m=loc.shape[-1], concentration=concentration)

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def _compute_kl(self, m, concentration):
        term1 = concentration * (
            ive(m / 2, concentration) / (ive(m / 2 - 1, concentration))
        )  # good

        term2 = (
            (m / 2 - 1) * concentration.log()
            - torch.tensor([2 * np.pi]).to(concentration.device).log() * (m / 2)
            - (ive(m / 2 - 1, concentration)).log()
            - concentration
        )  # good

        term3 = (
            -torch.lgamma(torch.tensor([m / 2]).to(concentration.device))
            + torch.tensor([2]).to(concentration.device).log()
            + torch.tensor([np.pi]).to(concentration.device).log() * (m / 2)
        )  # good

        return (term1 + term2 + term3).squeeze(-1)

    def _sample_von_mises(self, loc, concentration):

        # Generate uniformly on sphere
        v = torch.randn_like(loc[:, 1:])
        v = v / v.norm(dim=-1, keepdim=True)

        w = self._acc_rej_steps(m=loc.shape[-1], k=concentration)

        w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
        z = torch.cat((w, w_ * v), dim=-1)

        return self._householder_rotation(loc, z)

    def _householder_rotation(self, loc, z):
        e1 = torch.zeros(z.shape[-1]).to(z.device)
        e1[0] = 1
        u = e1 - loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)

        return z - 2 * u * (u * z).sum(dim=-1, keepdim=True)

    def _acc_rej_steps(self, m: int, k: torch.Tensor, device: str = "cpu"):

        batch_size = k.shape[0]

        c = torch.sqrt(4 * k ** 2 + (m - 1) ** 2)

        b = (-2 * k + c) / (m - 1)
        a = (m - 1 + 2 * k + c) / 4
        d = (4 * a * b) / (1 + b) - (m - 1) * np.log(m - 1)

        d.to(k.device)
        b.to(k.device)

        w = torch.zeros_like(k)

        stopping_mask = torch.ones_like(torch.tensor(b)).type(torch.bool)

        i = 0

        while stopping_mask.sum() > 0 and i < 100:

            i += 1

            eps = (
                dist.Beta(
                    torch.tensor(0.5 * (m - 1)).type(torch.float),
                    torch.tensor(0.5 * (m - 1)).type(torch.float),
                )
                .sample((batch_size, 1))
                .to(k.device)
            )

            w_ = (1 - (1 + b) * eps) / (1 - (1 - b) * eps)

            t = 2 * a * b / (1 - (1 - b) * eps)

            u = dist.Uniform(0, 1).sample((batch_size, 1)).to(k.device)

            acc = (m - 1) * t.log() - t + d > u.log()
            w[acc * stopping_mask] = w_[acc * stopping_mask]

            stopping_mask[acc * stopping_mask] = ~acc[acc * stopping_mask]

        return w

    def get_nll(self, data, n_samples=1, batch_size=100):
        """
        Function computed the estimate negative log-likelihood of the model. It uses importance
        sampling method with the approximate posterior distribution. This may take a while.

        Args:
            data (torch.Tensor): The input data from which the log-likelihood should be estimated.
                Data must be of shape [Batch x n_channels x ...]
            n_samples (int): The number of importance samples to use for estimation
            batch_size (int): The batchsize to use to avoid memory issues
        """

        if n_samples <= batch_size:
            n_full_batch = 1
        else:
            n_full_batch = n_samples // batch_size
            n_samples = batch_size

        log_p = []

        for i in range(len(data)):
            x = data[i].unsqueeze(0)

            log_p_x = []

            for j in range(n_full_batch):

                x_rep = torch.cat(batch_size * [x])

                encoder_output = self.encoder(x_rep)
                loc, log_concentration = (
                    encoder_output.embedding,
                    encoder_output.log_concentration,
                )

                # normalize mean
                loc = loc / loc.norm(dim=-1, keepdim=True)

                concentration = torch.nn.functional.softplus(log_concentration)
                z = self._sample_von_mises(loc, concentration)
                recon_x = self.decoder(z)["reconstruction"]
                m = loc.shape[-1]

                term1 = concentration * (loc * z).sum(dim=-1, keepdim=True)
                term2 = (
                    (m / 2 - 1) * concentration.log()
                    - torch.tensor([2 * np.pi]).to(concentration.device).log() * (m / 2)
                    - (ive(m / 2 - 1, concentration)).log()
                    - concentration
                )

                log_q_z_given_x = (term1 + term2).reshape(-1)  # VMF log-density

                log_p_z = -torch.ones_like(log_q_z_given_x) * (
                    -torch.lgamma(torch.tensor([m / 2]).to(concentration.device))
                    + torch.tensor([2]).to(concentration.device).log()
                    + torch.tensor([np.pi]).to(concentration.device).log() * (m / 2)
                )

                recon_x = self.decoder(z)["reconstruction"]

                if self.model_config.reconstruction_loss == "mse":

                    log_p_x_given_z = -0.5 * F.mse_loss(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1) - torch.tensor(
                        [np.prod(self.input_dim) / 2 * np.log(np.pi * 2)]
                    ).to(
                        data.device
                    )  # decoding distribution is assumed unit variance  N(mu, I)

                elif self.model_config.reconstruction_loss == "bce":

                    log_p_x_given_z = -F.binary_cross_entropy(
                        recon_x.reshape(x_rep.shape[0], -1),
                        x_rep.reshape(x_rep.shape[0], -1),
                        reduction="none",
                    ).sum(dim=-1)

                log_p_x.append(
                    log_p_x_given_z + log_p_z - log_q_z_given_x
                )  # log(2*pi) simplifies

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())

            if i % 100 == 0:
                print(f"Current nll at {i}: {np.mean(log_p)}")

        return np.mean(log_p)
