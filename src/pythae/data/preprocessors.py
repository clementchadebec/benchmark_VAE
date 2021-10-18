"""The purpose of the preprocessor is to ensure the data is not corrupted (no nan), reshape
it in case inconsistencies are detected, normalize it and converted it to a format handled by the
:class:`~pythae.trainers.Trainer`. In particular, an input data is converted to a
:class:`torch.Tensor` and all the data is gather into a :class:`~pythae.data.datastest.BaseDatset`
instance.

By choice, we do not provided very advanced preprocessing functions (such as image registrations)
since the augmentation method should be robust to huge differences in the data and be able to
reproduce and account for this diversity. More advanced preprocessing is up to the user.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from pythae.data.datasets import BaseDataset

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

HANDLED_NORMALIZATION = ["min_max_scaling", "individual_min_max_scaling", None]


class DataProcessor:
    """
    This is a basic class which preprocesses the data.
    Basically, it takes messy data, detects potential nan, bad types end convert the
    data to a type handled by the VAE models (*i.e.* `torch.Tensor`). Moreover, if the
    data does not have the same shape, a reshaping is applied and data is resized to the
    **minimal shape**.
    """

    def __init__(self):
        pass

    def process_data(self, data: Union[np.ndarray, torch.Tensor]) -> torch.tensor:
        """ This function detects potential check the data type, detects nan in input data and
        preprocessed the data so it can be handled by the models.

        Args:
            data (Union[np.ndarray, torch.Tensor]): The data that need to be
                checked. Expected:

                    - | np.ndarray of shape `num_data x n_channels x [optional depth] x 
                      | [optional height] x width x ...`
                    - | torch.Tensor of shape `num_data x n_channels x [optional depth] x
                      | [optional height] x width x ...`

        Returns:
            clean_data (torch.tensor): The data that has been cleaned
        """

        if isinstance(data, np.ndarray) or torch.is_tensor(data):
            data = self._process_data_array(data)

        else:
            raise TypeError(
                "Wrong data type provided. Expected one of "
                "[np.ndarray, torch.Tensor]"
            )

        return data

    @staticmethod
    def to_dataset(data: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """This method converts a set of ``torch.Tensor`` to a
        :class:`~pythae.data.datasets.BaseDataset`

        Args:
            data (torch.Tensor): The set of data as a big torch.Tensor
            labels (torch.Tensor): The targets labels as a big torch.Tensor

        Returns:
            (BaseDataset): The resulting dataset
        """

        if labels is None:
            labels = torch.ones(data.shape[0])

        labels = DataProcessor.to_tensor(labels)

        dataset = BaseDataset(data, labels)

        return dataset

    def _process_data_array(self, data: np.ndarray):

        # Detect potential nan
        if DataProcessor.has_nan(data):
            raise ValueError("Nan detected in input data!")

        data = DataProcessor.to_tensor(data)

        return data

    @staticmethod
    def to_tensor(data: np.ndarray) -> torch.Tensor:
        """Converts numpy arrays to `torch.Tensor` format

        Args:
            data (np.ndarray): The data to be converted

        Return:
            (torch.Tensor): The transformed data"""

        # check input type
        if not torch.is_tensor(data):
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    " Data must be either of type "
                    f"< 'torch.Tensor' > or < 'np.ndarray' > ({type(data)} provided). "
                    f" Check data"
                )

            else:
                try:
                    data = torch.tensor(data).type(torch.float)

                except (TypeError, RuntimeError) as e:
                    raise TypeError(
                        str(e.args) + ". Potential issues:\n"
                        "- input data has not the same shape in array\n"
                        "- input data with unhandable type"
                    ) from e

        return data

    @staticmethod
    def has_nan(data: torch.Tensor) -> bool:
        """Detects potential nan in input data

        Args:
            data (torch.Tensor): The data to be checked

        Return:
            (bool): True if data contains :obj:`nan`
        """

        if (data != data).sum() > 0:
            return True
        else:
            return False
