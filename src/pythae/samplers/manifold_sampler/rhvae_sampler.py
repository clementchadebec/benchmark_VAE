import torch

from ...models import RHVAE
from ..base import BaseSampler
from .rhvae_sampler_config import RHVAESamplerConfig


class RHVAESampler(BaseSampler):
    """Sampling form the inverse of the metric volume element of a :class:`~pythae.models.RHVAE`
    model.

    Args:
        model (RHVAE): The VAE model to sample from
        sampler_config (RHVAESamplerConfig): A RHVAESamplerConfig instance containing the main
            parameters of the sampler. If None, a pre-defined configuration is used. Default: None
    """

    def __init__(self, model: RHVAE, sampler_config: RHVAESamplerConfig = None):

        if sampler_config is None:
            sampler_config = RHVAESamplerConfig()

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

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

    def sample(
        self,
        num_samples: int = 1,
        batch_size: int = 500,
        output_dir: str = None,
        return_gen: bool = True,
        save_sampler_config: bool = False,
    ) -> torch.Tensor:
        """Main sampling function of the sampler.

        Args:
            num_samples (int): The number of samples to generate
            batch_size (int): The batch size to use during sampling
            output_dir (str): The directory where the images will be saved. If does not exist the
                folder is created. If None: the images are not saved. Defaults: None.
            return_gen (bool): Whether the sampler should directly return a tensor of generated
                data. Default: True.
            save_sampler_config (bool): Whether to save the sampler config. It is saved in
                output_dir

        Returns:
            ~torch.Tensor: The generated images
        """

        full_batch_nbr = int(num_samples / batch_size)
        last_batch_samples_nbr = num_samples % batch_size

        x_gen_list = []

        for i in range(full_batch_nbr):

            samples = self.hmc_sampling(batch_size)
            x_gen = self.model.decoder(z=samples)["reconstruction"].detach()

            if output_dir is not None:
                for j in range(batch_size):
                    self.save_img(
                        x_gen[j], output_dir, "%08d.png" % int(batch_size * i + j)
                    )

            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            samples = self.hmc_sampling(last_batch_samples_nbr)
            x_gen = self.model.decoder(z=samples)["reconstruction"].detach()

            if output_dir is not None:
                for j in range(last_batch_samples_nbr):
                    self.save_img(
                        x_gen[j],
                        output_dir,
                        "%08d.png" % int(batch_size * full_batch_nbr + j),
                    )

            x_gen_list.append(x_gen)

        if save_sampler_config:
            self.save(output_dir)

        if return_gen:
            return torch.cat(x_gen_list, dim=0)

    def hmc_sampling(self, n_samples: int):

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
