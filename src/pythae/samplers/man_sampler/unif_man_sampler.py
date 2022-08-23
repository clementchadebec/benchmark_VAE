import torch
import numpy as np
from torch.utils.data import DataLoader

from ...data.preprocessors import DataProcessor
from ...models import VAE
from ..base import BaseSampler
from .unif_man_sampler_config import UnifManVAESamplerConfig




class UnifManVAESampler(BaseSampler):
    """Fits a second VAE in the Autoencoder's latent space.

    Args:
        model (VAE): The VAE model to sample from
        sampler_config (TwoStageVAESamplerConfig): A TwoStageVAESamplerConfig instance containing
            the main parameters of the sampler. If None, a pre-defined configuration is used.
            Default: None

    .. note::

        The method :class:`~pythae.samplers.TwoStageVAESampler.fit` must be called to fit the
        sampler before sampling.
    """

    def __init__(self, model: VAE, sampler_config: UnifManVAESamplerConfig = None):

        assert issubclass(model.__class__, VAE), (
            "The UnifManVAESampler is only"
            f"applicable for VAE based models. Got {model.__class__}."
        )

        self.is_fitted = False

        if sampler_config is None:
            sampler_config = UnifManVAESamplerConfig()

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

        self.lbd = sampler_config.lbd
        self.tau = sampler_config.tau
        self.mcmc_steps = sampler_config.mcmc_steps
        self.eps_lf = sampler_config.eps_lf
        self.n_lf = sampler_config.n_lf
        self.n_medoids = sampler_config.n_medoids


    def fit(
        self, train_data, **kwargs
    ):
        """Method to fit the sampler from the training data

        Args:
            train_data (torch.Tensor): The train data needed to retreive the training embeddings
                    and fit the mixture in the latent space. Must be of shape n_imgs x im_channels x
                    ... and in range [0-1]
        """

        assert (
            train_data.max() >= 1 and train_data.min() >= 0
        ), "Train data must in the range [0-1]"

        dataset_type = (
            "DoubleBatchDataset"
            if self.model.model_name == "FactorVAE"
            else "BaseDataset"
        )

        data_processor = DataProcessor()
        train_data = data_processor.process_data(train_data).to(self.device)
        train_dataset = data_processor.to_dataset(train_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

        mu = []
        log_var = []

        try:
            with torch.no_grad():
                for _, inputs in enumerate(train_loader):
                    encoder_output = self.model.encoder(inputs['data'])
                    mu_ = encoder_output.embedding
                    log_var_ = encoder_output.log_covariance
                    mu.append(mu_)
                    log_var.append(log_var_)

        except RuntimeError:
            for _, inputs in enumerate(train_loader):
                encoder_output = self.model.encoder(inputs['data'])
                mu_ = encoder_output.embedding
                log_var_ = encoder_output.log_covariance
                mu.append(mu_)
                log_var.append(log_var_)

        mu = torch.cat(mu)
        log_var = torch.cat(log_var)

        print(self.n_medoids)
        medoids = mu[:self.n_medoids]
        centroids_idx = torch.arange(0, self.n_medoids).to(self.device)

        self.medoids = medoids
        self.centroids_idx = centroids_idx

        T = 0
        T_is = []
        for i in range(len(medoids)-1):
            #T_i = 1e10
            #for j in range(i+1, len(medoids)):
            mask = torch.tensor([k for k in range(len(medoids)) if k != i])
            dist = torch.norm(medoids[i].unsqueeze(0) - medoids[mask], dim=-1)
            #print(dist.shape)
            #if dist < T_i:
            T_i =torch.min(dist, dim=0)[0]
            T_is.append(T_i.item())

        T = np.max(T_is)
        print('Best temperature found: ', T)

        self._build_metrics(mu, log_var, centroids_idx, T=T, lbd=self.lbd)
        self.is_fitted = True

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

        if not self.is_fitted:
            raise ArithmeticError(
                "The sampler needs to be fitted by calling sampler.fit() method"
                "before sampling."
            )

        full_batch_nbr = int(num_samples / batch_size)
        last_batch_samples_nbr = num_samples % batch_size

        x_gen_list = []

        for i in range(full_batch_nbr):

            z, _ = self._hmc_sampling(batch_size)
            x_gen = self.model.decoder(z)["reconstruction"].detach()

            if output_dir is not None:
                for j in range(batch_size):
                    self.save_img(
                        x_gen[j], output_dir, "%08d.png" % int(batch_size * i + j)
                    )

            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            z = self._hmc_sampling(last_batch_samples_nbr)
            x_gen = self.model.decoder(z)["reconstruction"].detach()

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


    def _build_metrics(self, mu, log_var, idx=None, T=0.3, lbd=0.0001, tau=1e-10):

        if idx is not None:
            mu = mu[idx]
            log_var = log_var[idx]

        with torch.no_grad():
            self.model.M_i = torch.diag_embed((-log_var).exp()).detach()
            self.model.M_i_flat = (-log_var).exp().detach()
            self.model.M_i_inverse_flat = (log_var).exp().detach()
            self.model.centroids = mu.detach()
            self.model.T = T
            self.model.lbd = lbd
            self.model.tau = tau


            def G_sampl(z):
                omega = (
                    -(
                        torch.transpose(
                                    (self.model.centroids.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(-1), 2, 3) @ torch.diag_embed(self.model.M_i_flat).unsqueeze(0) @ (self.model.centroids.unsqueeze(0) -                                      z.unsqueeze(1)).unsqueeze(-1)
                                ) / self.model.T**2
                    ).exp()  

                #print("here", omega.shape, (torch.diag_embed(model.M_i_flat).unsqueeze(0) * omega
                #).sum(dim=1).shape, torch.diag_embed(torch.exp(-model.tau * torch.norm(z, dim=-1, keepdim=True)**2).repeat(1, model.latent_dim)).shape)      

                return (torch.diag_embed(self.model.M_i_flat).unsqueeze(0) * omega
                ).sum(dim=1) + self.model.lbd * torch.diag_embed(torch.exp(-self.model.tau * torch.norm(z, dim=-1, keepdim=True)**2).repeat(1, self.model.latent_dim))#torch.eye(self.model.latent_dim).to(device) * torch.exp(-self.model.tau * torch.norm(z, dim=-1, keepdim=True)**2)

            def G_interp(z):
                omega = (
                    -(
                        torch.transpose(
                                    (self.model.centroids.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(-1), 2, 3) @ torch.diag_embed(self.model.M_i_flat).unsqueeze(0) @ (self.model.centroids.unsqueeze(0) -                                      z.unsqueeze(1)).unsqueeze(-1)
                                ) / self.model.T**2
                    ).exp()

                return torch.inverse((torch.diag_embed(self.model.M_i_inverse_flat).unsqueeze(0) * omega
                ).sum(dim=1) + self.model.lbd * torch.eye(self.model.latent_dim).to(self.device))

            self.model.G_sampl = G_sampl
            self.model.G_interp = G_interp


    def _d_log_sqrt_det_G(self, z):
        with torch.no_grad():
            omega = (
                    -(
                        torch.transpose(
                                    (self.model.centroids.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(-1), 2, 3) @ self.model.M_i.unsqueeze(0) @ (self.model.centroids.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(-1)
                                ) / self.model.T**2
                    ).exp()
            
            d_omega_dz = ((-2 * self.model.M_i_flat * (z.unsqueeze(1) - self.model.centroids.unsqueeze(0)) / (self.model.T ** 2)).unsqueeze(-2) * omega).squeeze(-2) # good

            #print(d_omega_dz.shape, omega.shape)
        

            num = (d_omega_dz.unsqueeze(-2) * (self.model.M_i_flat.unsqueeze(0).unsqueeze(-1))).sum(1) # good
            
            denom = (self.model.M_i_flat.unsqueeze(0) * omega.squeeze(-1) + self.model.lbd).sum(1) # good
            #print((num / denom.unsqueeze(-1)))

            #print(num.shape, denom.shape)

        return torch.transpose(num / denom.unsqueeze(-1), 1, 2).sum(-1) - 2 * self.model.tau * z * torch.exp(-self.model.tau * torch.norm(z, dim=-1, keepdim=True)**2)# good!!!!


    def _log_pi(self, z):
        #print(model.G_sampl(z)[6],model.G_sampl(z)[6].det())
        return 0.5 * (torch.clamp(self.model.G_sampl(z).det(), 1e-10, 1e10)).log()


    def _hmc_sampling(self, n_samples=1):

        acc_nbr = torch.zeros(n_samples, 1).to(self.device)
        with torch.no_grad():

            idx = torch.randint(0, len(self.medoids), (n_samples,))

            #z0 = model.centroids[idx]

            z0 = self.medoids[idx]
    
            #z0 = torch.randn(n_samples, model.latent_dim).cuda()

            z = z0
            for i in range(self.mcmc_steps):
                #print(i)
                gamma = 2*torch.randn_like(z, device=self.device)
                rho = gamma# / self.beta_zero_sqrt

                H0 = -self._log_pi(z) + 0.5 * torch.norm(rho, dim=1) ** 2
                #print(H0)
                # print(model.G_inv(z).det())
                for k in range(self.n_lf):

                    #z = z.clone().detach().requires_grad_(True)
                    #log_det = G(z).det().log()

                    #g = torch.zeros(n_samples, model.latent_dim).cuda()
                    #for i in range(n_samples):
                    #    g[0] = -grad(log_det, z)[0][0]

                    g = -self._d_log_sqrt_det_G(z).reshape(
                        n_samples, self.model.latent_dim
                    )
                    # step 1
                    rho_ = rho - (self.eps_lf / 2) * g

                    # step 2
                    z = z + self.eps_lf * rho_

                    #z_ = z_.clone().detach().requires_grad_(True)
                    #log_det = 0.5 * G(z).det().log()
                    #log_det = G(z_).det().log()

                    #g = torch.zeros(n_samples, model.latent_dim).cuda()
                    #for i in range(n_samples):
                    #    g[0] = -grad(log_det, z_)[0][0]

                    g = -self._d_log_sqrt_det_G(z).reshape(
                        n_samples, self.model.latent_dim
                    )
                    #print(g)
                    # g = (Sigma_inv @ (z - mu).T).reshape(n_samples, 2)

                    # step 3
                    rho__ = rho_ - (self.eps_lf / 2) * g

                    # tempering
                    beta_sqrt = 1

                    rho =  rho__
                    #beta_sqrt_old = beta_sqrt

                H = -self._log_pi(z) + 0.5 * torch.norm(rho, dim=1) ** 2
                alpha = torch.exp(-H) / (torch.exp(-H0))
                #print(alpha)

                #print(-log_pi(best_model, z, best_model.G), 0.5 * torch.norm(rho, dim=1) ** 2)
                acc = torch.rand(n_samples).to(self.device)
                moves = (acc < alpha).type(torch.int).reshape(n_samples, 1)

                acc_nbr += moves

                z = z * moves + (1 - moves) * z0
                z0 = z
            #print(acc_nbr[:10])
            return z.detach()

