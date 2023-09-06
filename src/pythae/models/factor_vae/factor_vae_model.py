import os
from typing import Optional

import torch
import torch.nn.functional as F

from ...customexception import BadInheritanceError
from ...data.datasets import BaseDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseDiscriminator, BaseEncoder
from ..vae import VAE
from .factor_vae_config import FactorVAEConfig
from .factor_vae_utils import FactorVAEDiscriminator


class FactorVAE(VAE):
    """
    FactorVAE model.

    Args:
        model_config (FactorVAEConfig): The Variational Autoencoder configuration setting the main
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
        model_config: FactorVAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.discriminator = FactorVAEDiscriminator(latent_dim=model_config.latent_dim)

        self.model_name = "FactorVAE"
        self.gamma = model_config.gamma

    def set_discriminator(self, discriminator: BaseDiscriminator) -> None:
        r"""This method is called to set the discriminator network

        Args:
            discriminator (BaseDiscriminator): The discriminator module that needs to be set to the model.

        """
        if not issubclass(type(discriminator), BaseDiscriminator):
            raise BadInheritanceError(
                (
                    "Discriminator must inherit from BaseDiscriminator class from "
                    "pythae.models.base_architectures.BaseDiscriminator. Refer to documentation."
                )
            )

        self.discriminator = discriminator

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        x_in = inputs["data"]
        if x_in.shape[0] <= 1:
            raise ArithmeticError(
                "At least 2 samples in a batch are required for the `FactorVAE` model"
            )

        idx = torch.randperm(x_in.shape[0])
        idx_1 = idx[int(x_in.shape[0] / 2) :]
        idx_2 = idx[: int(x_in.shape[0] / 2)]

        # first batch
        x = inputs["data"][idx_1]

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        # second batch
        x_bis = inputs["data"][idx_2]

        encoder_output = self.encoder(x_bis)

        mu_bis, log_var_bis = encoder_output.embedding, encoder_output.log_covariance

        std_bis = torch.exp(0.5 * log_var_bis)
        z_bis, _ = self._sample_gauss(mu_bis, std_bis)

        z_bis_permuted = self._permute_dims(z_bis).detach()

        recon_loss, autoencoder_loss, discriminator_loss = self.loss_function(
            recon_x, x, mu, log_var, z, z_bis_permuted
        )

        loss = autoencoder_loss + discriminator_loss

        output = ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            autoencoder_loss=autoencoder_loss,
            discriminator_loss=discriminator_loss,
            recon_x=recon_x,
            recon_x_indices=idx_1,
            z=z,
            z_bis_permuted=z_bis_permuted,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z, z_bis_permuted):

        N = z.shape[0]  # batch size

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

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        latent_adversarial_score = self.discriminator(z)

        TC = (latent_adversarial_score[:, 0] - latent_adversarial_score[:, 1]).mean()
        autoencoder_loss = recon_loss + KLD + self.gamma * TC

        # discriminator loss
        permuted_latent_adversarial_score = self.discriminator(z_bis_permuted)

        true_labels = (
            torch.ones(z_bis_permuted.shape[0], requires_grad=False)
            .type(torch.LongTensor)
            .to(z.device)
        )
        fake_labels = (
            torch.zeros(z.shape[0], requires_grad=False)
            .type(torch.LongTensor)
            .to(z.device)
        )

        TC_permuted = F.cross_entropy(
            latent_adversarial_score, fake_labels
        ) + F.cross_entropy(permuted_latent_adversarial_score, true_labels)

        discriminator_loss = 0.5 * TC_permuted

        return (
            (recon_loss).mean(dim=0),
            (autoencoder_loss).mean(dim=0),
            (discriminator_loss).mean(dim=0),
        )

    def reconstruct(self, inputs: torch.Tensor):
        """This function returns the reconstructions of given input data.

        Args:
            inputs (torch.Tensor): The inputs data to be reconstructed of shape [B x input_dim]
            ending_inputs (torch.Tensor): The starting inputs in the interpolation of shape

        Returns:
            torch.Tensor: A tensor of shape [B x input_dim] containing the reconstructed samples.
        """
        encoder_output = self.encoder(inputs)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]
        return recon_x

    def interpolate(
        self,
        starting_inputs: torch.Tensor,
        ending_inputs: torch.Tensor,
        granularity: int = 10,
    ):
        """This function performs a linear interpolation in the latent space of the autoencoder
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

        encoder_output = self.encoder(starting_inputs)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        starting_z, _ = self._sample_gauss(mu, std)

        encoder_output = self.encoder(ending_inputs)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        ending_z, _ = self._sample_gauss(mu, std)

        t = torch.linspace(0, 1, granularity).to(starting_inputs.device)
        intep_line = (
            torch.kron(
                starting_z.reshape(starting_z.shape[0], -1), (1 - t).unsqueeze(-1)
            )
            + torch.kron(ending_z.reshape(ending_z.shape[0], -1), t.unsqueeze(-1))
        ).reshape((starting_z.shape[0] * t.shape[0],) + (starting_z.shape[1:]))

        decoded_line = self.decoder(intep_line).reconstruction.reshape(
            (
                starting_inputs.shape[0],
                t.shape[0],
            )
            + (starting_inputs.shape[1:])
        )

        return decoded_line

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _permute_dims(self, z):
        permuted = torch.zeros_like(z)

        for i in range(z.shape[-1]):
            perms = torch.randperm(z.shape[0]).to(z.device)
            permuted[:, i] = z[perms, i]

        return permuted
