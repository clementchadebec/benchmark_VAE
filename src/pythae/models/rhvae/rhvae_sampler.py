import datetime
import logging
import os

import torch

from ..base.base_sampler import BaseSampler
from .rhvae_config import RHVAESamplerConfig
from .rhvae_model import RHVAE

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class RHVAESampler(BaseSampler):
    """Hamiltonian Monte Carlo Sampler class.
    This is an implementation of the Hamiltonian/Hybrid Monte Carlo sampler
    (https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)

    Args:
        model (RHVAE): The VAE model to sample from
        sampler_config (RHVAESamplerConfig): A HMCSamplerConfig instance containing the main
            parameters of the sampler. If None, a pre-defined configuration is used. Default: None
    """

    def __init__(self, model: RHVAE, sampler_config: RHVAESamplerConfig = None):

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

        self.sampler_config = sampler_config

        self.model.M_tens = self.model.M_tens.to(self.device)
        self.model.centroids_tens = self.model.centroids_tens.to(self.device)

        self.mcmc_steps_nbr = sampler_config.mcmc_steps_nbr
        self.n_lf = torch.tensor([sampler_config.n_lf]).to(self.device)
        self.eps_lf = torch.tensor([sampler_config.eps_lf]).to(self.device)
        self.beta_zero_sqrt = (
            torch.tensor([sampler_config.beta_zero]).to(self.device).sqrt()
        )

        self.log_pi = RHVAESampler.log_sqrt_det_G_inv
        self.grad_func = RHVAESampler.grad_log_prop

    def sample(self, samples_number):
        """
        HMC sampling with a RHVAE.

        The data is saved in the ``output_dir`` (folder passed in the
        :class:`~pyraug.models.base.base_config.BaseSamplerConfig` instance) in a folder named
        ``generation_YYYY-MM-DD_hh-mm-ss``. If ``output_dir`` is None, a folder named
        ``dummy_output_dir`` is created in this folder.

        Args:
            num_samples (int): The number of samples to generate
        """

        assert samples_number > 0, "Provide a number of samples > 0"

        self._sampling_signature = (
            str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
        )

        sampling_dir = os.path.join(
            self.sampler_config.output_dir, f"generation_{self._sampling_signature}"
        )

        if not os.path.exists(sampling_dir):
            os.makedirs(sampling_dir)
            logger.info(
                f"Created {sampling_dir}. "
                "Generated data and sampler config will be saved here.\n"
            )

        full_batch_nbr = int(samples_number / self.sampler_config.batch_size)
        last_batch_samples_nbr = samples_number % self.sampler_config.batch_size

        generated_data = []

        file_count = 0
        data_count = 0

        logger.info("Generation successfully launched !\n")

        for i in range(full_batch_nbr):

            samples = self.hmc_sampling(self.batch_size)
            x_gen = self.model.decoder(z=samples).detach()
            assert len(x_gen.shape) == 2
            generated_data.append(x_gen)
            data_count += self.batch_size

            while data_count >= self.samples_per_save:
                self.save_data_batch(
                    data=torch.cat(generated_data)[: self.samples_per_save],
                    dir_path=sampling_dir,
                    number_of_samples=self.samples_per_save,
                    batch_idx=file_count,
                )

                file_count += 1
                data_count -= self.samples_per_save
                generated_data = list(
                    torch.cat(generated_data)[self.samples_per_save :].unsqueeze(0)
                )

        if last_batch_samples_nbr > 0:
            samples = self.hmc_sampling(last_batch_samples_nbr)
            x_gen = self.model.decoder(z=samples).detach()
            generated_data.append(x_gen)

            data_count += last_batch_samples_nbr

            while data_count >= self.samples_per_save:
                self.save_data_batch(
                    data=torch.cat(generated_data)[: self.samples_per_save],
                    dir_path=sampling_dir,
                    number_of_samples=self.samples_per_save,
                    batch_idx=file_count,
                )

                file_count += 1
                data_count -= self.samples_per_save
                generated_data = list(
                    torch.cat(generated_data)[self.samples_per_save :].unsqueeze(0)
                )

        if data_count > 0:
            self.save_data_batch(
                data=torch.cat(generated_data),
                dir_path=sampling_dir,
                number_of_samples=data_count,
                batch_idx=file_count,
            )

        self.save(sampling_dir)

    def hmc_sampling(self, n_samples):

        with torch.no_grad():

            idx = torch.randint(len(self.model.centroids_tens), (n_samples,))

            z0 = self.model.centroids_tens[idx]

            beta_sqrt_old = self.beta_zero_sqrt
            z = z0
            for i in range(self.mcmc_steps_nbr):

                gamma = torch.randn_like(z, device=self.device)
                rho = gamma / self.beta_zero_sqrt

                H0 = -self.log_pi(z, self.model) + 0.5 * torch.norm(rho, dim=1) ** 2
                # print(model.G_inv(z).det())

                for k in range(self.n_lf):

                    g = -self.grad_func(z, self.model).reshape(
                        n_samples, self.model.latent_dim
                    )
                    # step 1
                    rho_ = rho - (self.eps_lf / 2) * g

                    # step 2
                    z = z + self.eps_lf * rho_

                    g = -self.grad_func(z, self.model).reshape(
                        n_samples, self.model.latent_dim
                    )
                    # g = (Sigma_inv @ (z - mu).T).reshape(n_samples, 2)

                    # step 3
                    rho__ = rho_ - (self.eps_lf / 2) * g

                    # tempering
                    beta_sqrt = RHVAESampler.tempering(
                        k + 1, self.n_lf, self.beta_zero_sqrt
                    )

                    rho = (beta_sqrt_old / beta_sqrt) * rho__
                    beta_sqrt_old = beta_sqrt

                H = -self.log_pi(z, self.model) + 0.5 * torch.norm(rho, dim=1) ** 2
                alpha = torch.exp(-H) / (torch.exp(-H0))
                acc = torch.rand(n_samples).to(self.device)
                moves = (acc < alpha).type(torch.int).reshape(n_samples, 1)

                z = z * moves + (1 - moves) * z0

                z0 = z

            return z

    @staticmethod
    def tempering(k, K, beta_zero_sqrt):
        beta_k = ((1 - 1 / beta_zero_sqrt) * (k / K) ** 2) + 1 / beta_zero_sqrt

        return 1 / beta_k

    @staticmethod
    def log_sqrt_det_G_inv(z, model):
        return torch.log(torch.sqrt(torch.det(model.G_inv(z))) + 1e-10)

    @staticmethod
    def grad_log_sqrt_det_G_inv(z, model):
        return (
            -0.5
            * torch.transpose(model.G(z), 1, 2)
            @ torch.transpose(
                (
                    -2
                    / (model.temperature ** 2)
                    * (model.centroids_tens.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(2)
                    @ (
                        model.M_tens.unsqueeze(0)
                        * torch.exp(
                            -torch.norm(
                                model.centroids_tens.unsqueeze(0) - z.unsqueeze(1),
                                dim=-1,
                            )
                            ** 2
                            / (model.temperature ** 2)
                        )
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                ).sum(dim=1),
                1,
                2,
            )
        )

    @staticmethod
    def grad_log_prop(z, model):
        def grad_func(z, model):
            return RHVAESampler.grad_log_sqrt_det_G_inv(z, model)

        return grad_func(z, model)
