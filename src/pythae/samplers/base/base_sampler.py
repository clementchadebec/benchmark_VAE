import logging
import os
from typing import Any, Dict

import numpy as np
import torch
from imageio import imwrite

from ...models import BaseAE
from .base_sampler_config import BaseSamplerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseSampler:
    """Base class for samplers used to generate from the VAEs models.

    Args:
        model (BaseAE): The vae model to sample from.
        sampler_config (BaseSamplerConfig): An instance of BaseSamplerConfig in which any sampler's
            parameters is made available. If None a default configuration is used. Default: None
    """

    def __init__(self, model: BaseAE, sampler_config: BaseSamplerConfig = None):

        if sampler_config is None:
            sampler_config = BaseSamplerConfig()

        self.model = model
        self.model.eval()
        self.sampler_config = sampler_config
        self.is_fitted = False

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model.device = device

        self.model.to(device)

    def fit(self, *args, **kwargs):
        """Function to be called to fit the sampler before sampling"""
        pass

    def sample(
        self,
        num_samples: int = 1,
        batch_size: int = 500,
        output_dir: str = None,
        return_gen: bool = True,
    ):
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
            print(f"--> Created folder {dir_path}. Images will be saved here")

        img = 255.0 * torch.movedim(img_tensor, 0, 2).cpu().detach().numpy()
        if img.shape[-1] == 1:
            img = np.repeat(img, repeats=3, axis=-1)

        img = img.astype("uint8")
        imwrite(os.path.join(dir_path, f"{img_name}"), img)

    def _set_inputs_to_device(self, inputs: Dict[str, Any]):

        inputs_on_device = inputs

        if self.device == "cuda":
            cuda_inputs = dict.fromkeys(inputs)

            for key in inputs.keys():
                if torch.is_tensor(inputs[key]):
                    cuda_inputs[key] = inputs[key].cuda()

                else:
                    cuda_inputs[key] = inputs[key]
            inputs_on_device = cuda_inputs

        return inputs_on_device
