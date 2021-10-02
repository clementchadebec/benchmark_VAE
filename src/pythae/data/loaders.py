"""The loaders are used to load the data from a particular format to :class:`numpy.ndarray` or
:class:`List[numpy.ndarray]`
"""


import os
from typing import List, Union

import nibabel as nib
import numpy as np
import torch
from PIL import Image

HANDLED_TYPES = [".pt", ".nii", "nii.gz", "bmp", "jpg", "jpeg", "png"]


class BaseDataGetter:
    """This is the Base data loader from which all future loaders must inherit.
    """

    @classmethod
    def load(cls):
        raise NotImplementedError()


class ImageGetterFromFolder(BaseDataGetter):
    """This loader allows you to load imagining data directly from a folder and convert it to
    :class:`np.ndarray`. The data must be all located in a folder where each file is an image.

    Handled types are ('.pt', '.nii', 'nii.gz', 'bmp', 'jpg', 'jpeg', 'png')
    """

    @classmethod
    def load(cls, dir_path: str) -> List[np.ndarray]:

        im_files = os.listdir(dir_path)

        data_list = []

        for im_name in im_files:
            im_path = os.path.join(dir_path, im_name)
            im = cls.load_image(im_path)

            data_list.append(im)

        return data_list

    @classmethod
    def load_image(cls, im_path: Union[str, os.PathLike]) -> np.array:
        """Loads an image and returns an array.

        Handled types are ('.pt', '.nii', 'nii.gz', 'bmp', 'jpg', 'jpeg', 'png')

        Args:
            im_path (str, os.Pathlike): The path to the image

        Returns:
            (np.array): The loaded image of shape n_channels x [optional depth] x height x width
        """

        if not os.path.isfile(im_path):
            raise FileNotFoundError(f"The file {im_path} does not exist")

        if not cls.is_handled_file(im_path):
            raise TypeError(
                f"Image type '{im_path.split('.')[-1]}' not handled. Extensions handled"
                f" {HANDLED_TYPES}"
            )

        if im_path.endswith((".nii", "nii.gz")):
            data = cls._from_nifti(im_path)

        elif im_path.endswith(".pt"):
            data = torch.load(im_path)
            if torch.is_tensor(data):
                try:
                    data = data.numpy()

                except Exception as e:
                    raise e

            else:
                assert isinstance(data, np.ndarray), (
                    "Only np.ndarray and torch.Tensor can be "
                    "loaded from a '.pt' file."
                )

        else:
            im = Image.open(im_path).convert("RGB")
            data = np.array(im)

            # set channel first
            data = np.moveaxis(data, 2, 0)

        return data.astype(np.float64)

    @classmethod
    def is_handled_file(cls, im_path: Union[str, os.PathLike]) -> bool:
        """Checks if the path provided leads to an handable file

        Args:
            im_path (str, os.Pathlike): The path to the image

        Returns:
            (bool): If True, the file exists and is an handable file
        """

        return im_path.endswith(tuple(HANDLED_TYPES))

    @classmethod
    def _from_nifti(cls, im_path):
        img = nib.load(im_path)
        data = img.get_fdata()

        return data
