import logging
import os
import numpy as np

import torch
from imageio import imwrite

from .base_sampler_config import BaseSamplerConfig
from ...models import BaseAE

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseSampler:
    """Base class for sampler used to generate from the VAEs models

    Args:
        model (BaseAE): The vae model to sample from.
        sampler_config (BaseSamplerConfig): An instance of BaseSamplerConfig in which any sampler's
            parameters is made available. If None a default configuration is used. Default: None
    """

    def __init__(self, model: BaseAE, sampler_config: BaseSamplerConfig = None):

        if sampler_config is None:
            sampler_config = BaseSamplerConfig()


        self.model = model
        self.sampler_config = sampler_config

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)

    def fit(self, *args, **kwargs):
        """Function to be called to fit the sampler before sampling
        """
        pass

    def sample(
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
        raise NotImplementedError()

    def save(self, dir_path):
        """Method to save the sampler config. The config is saved a as ``sampler_config.json``
        file in ``dir_path``"""

        self.sampler_config.save_json(dir_path, "sampler_config")

    def save_img(self, img_tensor: torch.Tensor, dir_path: str, img_name: str):
        """Saves a data point as .png file in dir_path with img_name as name.

        Args:
            img_tensor (torch.Tensor): The image of shape CxHxW in the range [0-1]
            dir_path (str): The folder where in which the images must be saved
            ig_name (str): The name to apply to the file containing the image.
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"--> Created folder {dir_path}. Image will be saved here")        

        img = 255. * torch.movedim(img_tensor, 0, 2).cpu().detach().numpy()
        if img.shape[-1]==1:
            img = np.repeat(img, repeats=3, axis=-1)
        
        img = img.astype('uint8')
        imwrite(os.path.join(dir_path, f"{img_name}.png"), img)







    #def save_data_batch(self, data, dir_path, number_of_samples, batch_idx):
    #    """
    #    Method to save a batch of generated data. The data will be saved in the
    #    ``dir_path`` folder. The batch of data
    #    is saved in a file named ``generated_data_{number_of_samples}_{batch_idx}.pt``
#
    #    Args:
    #        data (torch.Tensor): The data to save
    #        dir_path (str): The folder where the data and config file must be saved
    #        batch_idx (int): The batch idx
#
    #    .. note::
    #        You can then easily reload the generated data using
#
    #        .. code-block:
#
    #            >>> import torch
    #            >>> import os
    #            >>> data = torch.load(
    #            ...    os.path.join(
    #            ...        'dir_path', 'generated_data_{number_of_samples}_{batch_idx}.pt'))
    #    """
#
    #    if not os.path.exists(dir_path):
    #        os.makedirs(dir_path)
#
    #    torch.save(
    #        data,
    #        os.path.join(
    #            dir_path, f"generated_data_{number_of_samples}_{batch_idx}.pt"
    #        ),
    #    )
#