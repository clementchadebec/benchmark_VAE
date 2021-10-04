import torch

from ...models import BaseAE
from ...samplers import BaseSampler, BaseSamplerConfig

class NormalSampler(BaseSampler):
    """Basic sampler sampling from a N(0, 1) in the Autoencoder's latent space

    Args:
        model (BaseAE): The vae model to sample from.
        sampler_config (BaseSamplerConfig): An instance of BaseSamplerConfig in which any sampler's
            parameters is made available. If None a default configuration is used. Default: None

    """


    def __init__(self, model: BaseAE, sampler_config: BaseSamplerConfig=None):

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

        
    def sample(self, num_samples: int) -> torch.Tensor:
        self,
        num_samples: int=1,
        batch_size: int = 500,
        output_dir:str=None,
        return_gen: bool=True):
        """Main sampling function of the sampler.

        Args:
            num_samples (int): The number of samples to generate
            batch_size (int): The batch size to use during sampling
            output_dir (str): The directory where the images will be saved. If does not exist the 
                folder is created. If None: the images are not saved. Defaults: None.
            return_gen (bool): Whether the sampler should directly return a tensor of generated 
                data. Default: True.
        
        Returns:
            ~torch.Tensor: The generated images
        """
        
        full_batch_nbr = int(num_samples / self.sampler_config.batch_size)
        last_batch_samples_nbr = num_samples % self.sampler_config.batch_size

        x_gen_list = []

        for i in range(full_batch_nbr):
            z = torch.randn(num_samples).to(self.device)
            x_gen = self.model.decoder(z)
            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            z = torch.randn(last_batch_samples_nbr).to(self.device)
            x_gen = self.model.decoder(z)
            x_gen_list.append(x_gen)


        return torch.cat(x_gen)