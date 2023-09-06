import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_SVAE_MLP, Encoder_VAE_MLP
from ..vae import VAE
from .pvae_config import PoincareVAEConfig
from .pvae_utils import PoincareBall, RiemannianNormal, WrappedNormal


class PoincareVAE(VAE):
    """Poincaré Variational Autoencoder model.

    Args:
        model_config (PoincareVAEConfig): The Poincaré Variational Autoencoder configuration
            setting the main parameters of the model.

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
        model_config: PoincareVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "PoincareVAE"

        self.latent_manifold = PoincareBall(
            dim=model_config.latent_dim, c=model_config.curvature
        )

        if model_config.prior_distribution == "riemannian_normal":
            self.prior = RiemannianNormal
        else:
            self.prior = WrappedNormal

        if model_config.posterior_distribution == "riemannian_normal":
            warnings.warn(
                "Carefull, this model expects the encoder to give a one dimensional "
                "`log_concentration` tensor for the Riemannian normal distribution. "
                "Make sure the encoder actually outputs this."
            )
            self.posterior = RiemannianNormal
        else:
            self.posterior = WrappedNormal

        if encoder is None:
            if model_config.posterior_distribution == "riemannian_normal":
                encoder = Encoder_SVAE_MLP(model_config)
            else:
                encoder = Encoder_VAE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

        self._pz_mu = nn.Parameter(
            torch.zeros(1, model_config.latent_dim), requires_grad=False
        )
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=False)

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

        if self.model_config.posterior_distribution == "riemannian_normal":
            mu, log_var = encoder_output.embedding, encoder_output.log_concentration
        else:
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)

        qz_x = self.posterior(loc=mu, scale=std, manifold=self.latent_manifold)
        z = qz_x.rsample(torch.Size([1]))

        recon_x = self.decoder(z.squeeze(0))["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, z, qz_x)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z.squeeze(0),
        )

        return output

    def loss_function(self, recon_x, x, z, qz_x):

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

        pz = self.prior(
            loc=self._pz_mu, scale=self._pz_logvar.exp(), manifold=self.latent_manifold
        )

        KLD = (qz_x.log_prob(z) - pz.log_prob(z)).sum(-1).squeeze(0)

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

    def interpolate(
        self,
        starting_inputs: torch.Tensor,
        ending_inputs: torch.Tensor,
        granularity: int = 10,
    ):
        """This function performs a geodesic interpolation in the poincaré disk of the autoencoder
        from starting inputs to ending inputs. It returns the interpolation trajectories.

        Args:
            starting_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            ending_inputs (torch.Tensor): The starting inputs in the interpolation of shape
                [B x input_dim]
            granularity (int): The granularity of the interpolation.

        Returns:
            torch.Tensor: A tensor of shape [B x granularity x input_dim] containing the
            interpolation trajectories.
        """
        assert starting_inputs.shape[0] == ending_inputs.shape[0], (
            "The number of starting_inputs should equal the number of ending_inputs. Got "
            f"{starting_inputs.shape[0]} sampler for starting_inputs and {ending_inputs.shape[0]} "
            "for endinging_inputs."
        )

        starting_z = self.encoder(starting_inputs).embedding
        ending_z = self.encoder(ending_inputs).embedding
        t = torch.linspace(0, 1, granularity).to(starting_inputs.device)

        inter_geo = torch.zeros(
            starting_inputs.shape[0], granularity, starting_z.shape[-1]
        ).to(starting_z.device)

        for i, t_i in enumerate(t):
            z_i = self.latent_manifold.geodesic(t_i, starting_z, ending_z)
            inter_geo[:, i, :] = z_i

        decoded_geo = self.decoder(
            inter_geo.reshape(
                (starting_z.shape[0] * t.shape[0],) + (starting_z.shape[1:])
            )
        ).reconstruction.reshape(
            (
                starting_inputs.shape[0],
                t.shape[0],
            )
            + (starting_inputs.shape[1:])
        )
        return decoded_geo

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
                if self.model_config.posterior_distribution == "riemannian_normal":
                    mu, log_var = (
                        encoder_output.embedding,
                        encoder_output.log_concentration,
                    )
                else:
                    mu, log_var = (
                        encoder_output.embedding,
                        encoder_output.log_covariance,
                    )

                std = torch.exp(0.5 * log_var)

                qz_x = self.posterior(loc=mu, scale=std, manifold=self.latent_manifold)
                z = qz_x.rsample(torch.Size([1]))

                pz = self.prior(
                    loc=self._pz_mu,
                    scale=self._pz_logvar.exp(),
                    manifold=self.latent_manifold,
                )

                log_q_z_given_x = qz_x.log_prob(z).sum(-1).squeeze(0)
                log_p_z = pz.log_prob(z).sum(-1).squeeze(0)

                recon_x = self.decoder(z.squeeze(0))["reconstruction"]

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

                log_p_x.append(log_p_x_given_z + log_p_z - log_q_z_given_x)

            log_p_x = torch.cat(log_p_x)

            log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
        return np.mean(log_p)
