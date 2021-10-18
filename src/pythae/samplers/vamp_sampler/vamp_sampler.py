import torch

from ...models import VAMP
from ...samplers import BaseSampler, BaseSamplerConfig

class VAMPSampler(BaseSampler):
    """Sampling from the VAMP prior

    Args:
        model (VAMP): The vae model to sample from.
        sampler_config (BaseSamplerConfig): An instance of BaseSamplerConfig in which any sampler's
            parameters is made available. If None a default configuration is used. Default: None

    """


    def __init__(self, model: VAMP, sampler_config: BaseSamplerConfig=None):

        assert isinstance(model, VAMP),  'This sampler is only suitable for VAMP model'

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

        
    def sample(
        self,
        num_samples: int=1,
        batch_size: int = 500,
        output_dir:str=None,
        return_gen: bool=True,
        save_sampler_config: bool=False
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

        batch_size = min(self.model.idle_input.shape[0], batch_size)

        full_batch_nbr = int(num_samples / batch_size)
        last_batch_samples_nbr = num_samples % batch_size

        x_gen_list = []

        for i in range(full_batch_nbr):
            means = self.model.pseudo_inputs(self.model.idle_input.to(self.device))[:batch_size]
            
            encoder_output = self.model.encoder(means)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std

            x_gen = self.model.decoder(z)['reconstruction'].detach()

            if output_dir is not None:
                for j in range(batch_size):
                    self.save_img(x_gen[j], output_dir, '%08d.png' % int(batch_size*i + j))


            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            means = self.model.pseudo_inputs(self.model.idle_input.to(
                self.device))[:last_batch_samples_nbr].to(self.device)
            
            encoder_output = self.model.encoder(means)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std

            x_gen = self.model.decoder(z)['reconstruction'].detach()

            if output_dir is not None:
                for j in range(last_batch_samples_nbr):
                    self.save_img(
                        x_gen[j], output_dir, '%08d.png' % int(batch_size*full_batch_nbr + j))


            x_gen_list.append(x_gen)

        if save_sampler_config:
            self.save(output_dir)

        if return_gen:
            return torch.cat(x_gen_list, dim=0)