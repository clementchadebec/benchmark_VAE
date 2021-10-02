"""The purpose of the preprocessor is to ensure the data is not corrupted (no nan), reshape
it in case inconsistencies are detected, normalize it and converted it to a format handled by the
:class:`~pyraug.trainers.Trainer`. In particular, an input data is converted to a
:class:`torch.Tensor` and all the data is gather into a :class:`~pyraug.data.datastest.BaseDatset`
instance.

By choice, we do not provided very advanced preprocessing functions (such as image registrations)
since the augmentation method should be robust to huge differences in the data and be able to
reproduce and account for this diversity. More advanced preprocessing is up to the user.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from pyraug.data.datasets import BaseDataset

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

    def __init__(self, data_normalization_type="min_max_scaling"):

        if data_normalization_type not in HANDLED_NORMALIZATION:
            raise RuntimeError(
                "Wrong normalization type provided. "
                f"Handled: {HANDLED_NORMALIZATION}. Check doc for further details."
            )

        self.data_normalization = data_normalization_type

    def process_data(
        self,
        data: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
    ) -> torch.tensor:
        """ This function detects potential check the data type, detects nan in input data and
        preprocessed the data so it can be handled by the models.

        Args:
            data (Union[np.ndarray, torch.Tensor, List[np.ndarray]]): The data that need to be
                checked. Expected:

                    - list of `np.ndarray` or
                    `torch.Tensor` of shape: `n_channels x [optional depth] x [optional height] x width`
                    - np.ndarray of shape `num_data x n_channels x [optional depth] x [optional height] x width`
                    - torch.Tensor of shape `num_data x n_channels x [optional depth] x [optional height] x width`

        Returns:
            clean_data (torch.tensor): The data that has bee cleaned

        .. warning::
            If you set ``normalized_data`` to False because you applied your own preprocessing for
            example, you must ensure that your data is comprised between 0 and 1 or an exception
            will be raised.

        .. note::
            If you provide input data that has different shapes (e.g. images of shape (3, 20, 30)
            and (3, 15, 60)) the data is reshaped to the minimum shape `i.e.` shape of (3, 15, 30).
        """

        if isinstance(data, list):
            for d in data:
                if not isinstance(d, np.ndarray) and not torch.is_tensor(d):
                    raise TypeError(
                        "Unhandled type in provided data. Expect list of 'np.ndarray' "
                        f"or 'torch.Tensor' but got: {type(d)}"
                    )

            data = self._process_data_list(data)

        elif isinstance(data, np.ndarray) or torch.is_tensor(data):
            data = self._process_data_array(data)

        else:
            raise TypeError(
                "Wrong data type provided. Expected one of "
                "[np.ndarray, torch.Tensor, List[np.ndarray, torch.Tensor]]"
            )

        return data

    @staticmethod
    def to_dataset(data: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """This method converts a set of ``torch.Tensor`` to a
        :class:`~pyraug.data.datasets.BaseDataset`

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

        # normalized each data point
        if self.data_normalization is not None:
            data = self.normalize_data(data)
            logger.info(f"Data normalized using {self.data_normalization}.")
            logger.info(
                " -> If this is not the desired behavior pass an instance of DataProcess "
                "with 'data_normalization_type' attribute set to desired normalization or None\n"
            )

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

    @staticmethod
    def _individual_min_max_scaling(data: torch.Tensor) -> torch.Tensor:

        d = data.reshape(data.shape[0], -1)

        clean_data = (d - d.min(dim=-1).values.unsqueeze(-1)) / (
            d.max(dim=-1).values.unsqueeze(-1) - d.min(dim=-1).values.unsqueeze(-1)
        )

        return clean_data.reshape_as(data)

    @staticmethod
    def _min_max_scaling(data: torch.Tensor) -> torch.Tensor:

        clean_data = (data - data.min().unsqueeze(-1)) / (
            data.max().unsqueeze(-1) - data.min().unsqueeze(-1)
        )

        return clean_data

    def normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        This function normalizes the input data so that all the values are between 0 and 1

        Args:
            data (torch.Tensor): The data to normalize

        Retruns:
            (torch.Tensor): The normalized data
        """

        if self.data_normalization == "min_max_scaling":
            clean_data = DataProcessor._min_max_scaling(data)

        elif self.data_normalization == "individual_min_max_scaling":
            clean_data = DataProcessor._individual_min_max_scaling(data)

        else:
            clean_data = data

        return clean_data

    def _process_data_list(
        self, data_list: List[Union[torch.Tensor, np.ndarray]]
    ) -> torch.tensor:

        tensor_list = []
        min_shape = ()
        diff_shape_count = 0
        for data in data_list:

            # Detect potential nan
            if DataProcessor.has_nan(data):
                raise ValueError("Nan detected in input data!")

            data = DataProcessor.to_tensor(data)

            data = data.unsqueeze(0)

            if min_shape == ():
                min_shape = tuple(data.shape)

            elif data.shape != min_shape:
                # count the number of data having a shape different from the min
                diff_shape_count += 1

                if len(data.shape) != len(min_shape):
                    raise RuntimeError(
                        f"Found {len(data.shape)}D and {len(min_shape)}D images ! "
                        "Ensure all images have same shape"
                    )

                else:
                    new_min_shape = list(min_shape)
                    for (i, shape_i) in enumerate(data.shape):
                        if shape_i < min_shape[i]:
                            new_min_shape[i] = shape_i
                        else:
                            new_min_shape[i] = min_shape[i]

                    min_shape = tuple(new_min_shape)

            tensor_list.append(data)

        # reshape if data has not the same shape
        if diff_shape_count >= 1:
            data = DataProcessor.reshape_data(tensor_list, min_shape)
            logger.info("Data of different shapes detected !")
            logger.info(
                f" -> Reshaped data to minimum size (data of shape: {tuple(data.shape)}).\n"
            )

        else:
            data = torch.cat(tensor_list)

        # normalized each data point
        if self.data_normalization is not None:
            data = self.normalize_data(data)
            logger.info(f"Data normalized using {self.data_normalization}.")
            logger.info(
                " -> If this is not the desired behavior pass an instance of DataProcess "
                "with 'data_normalization_type' attribute set to desired normalization or None\n"
            )

        if data.min() < 0 or data.max() > 1:
            raise RuntimeError(
                "Data must be normalized between 0 and 1. You can change "
                "'data_normalization_type' attributes to automatically normalized the data "
                "if this is the desired behavior."
            )

        return data

    @staticmethod
    def reshape_data(
        data_list: List[torch.Tensor], target_shape: Tuple[int, ...]
    ) -> torch.Tensor:

        """
        This function takes an input data and reshape it to a target shape

        Args:
            data (torch.Tensor): The data to reshape.
                Expected shape:
                `mini_batch x n_channels x [optional depth] x [optional height] x width`
        """

        tensor_list = []

        for data in data_list:

            if data.shape != target_shape:
                data = DataProcessor._reshape_data(data, target_shape[2:])
            tensor_list.append(data)

        data = torch.cat(tensor_list)

        return data

    @staticmethod
    def _reshape_data(
        data: torch.Tensor, target_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        This function takes an input data and reshape it to a target shape

        Args:
            data (torch.Tensor): The data to reshape.
                Expected shape:
                `mini_batch x n_channels x [optional depth] x [optional height] x width`
        """

        if len(data.shape) == 3:
            data = torch.nn.functional.interpolate(
                input=data, size=target_shape, mode="linear", align_corners=False
            )

        elif len(data.shape) == 4:
            data = torch.nn.functional.interpolate(
                input=data, size=target_shape, mode="bilinear", align_corners=False
            )

        elif len(data.shape) == 5:
            data = torch.nn.functional.interpolate(
                input=data, size=target_shape, mode="trilinear", align_corners=False
            )

        else:
            raise NotImplementedError(
                "Reshaping only implemented for 3D, 4D and 5D data"
            )

        return data
