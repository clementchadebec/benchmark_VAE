"""The pythae's Datasets inherit from
:class:`torch.utils.data.Dataset` and must be used to convert the data before
training. As of today, it only contains the :class:`pythae.data.BaseDatset` useful to train a
VAE model but other Datatsets will be added as models are added.
"""
import torch
from torch.utils.data import Dataset
from typing import Any, Tuple
from collections import OrderedDict


class DatasetOutput(OrderedDict):
    """Base DatasetOutput class fixing the output type from the dataset. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())

class BaseDataset(Dataset):
    """This class is the Base class for pythae's dataset

    A ``__getitem__`` is redefined and outputs a python dictionnary
    with the keys corresponding to `data` and `labels`.
    This Class should be used for any new data sets.
    """

    def __init__(self, data, labels):

        self.labels = labels.type(torch.float)
        self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionnary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
        """
        # Select sample
        X = self.data[index]
        y = self.labels[index]

        return DatasetOutput(
            data=X,
            labels=y
        )


class DoubleBatchDataset(BaseDataset):
    """This class is Dataset inheriting from :class:`pythae.data. instance outputing two different sets of tenosr at the same time.
    This is for instance needed in the :class:`pythae.models.FactorVAE` model.

    A ``__getitem__`` is redefined and outputs a python dictionnary
    with the keys corresponding to `data`, `data_bis` and `labels`
    This Class should be used for any new data sets.
    """

    def __init__(self, data, labels):

        self.labels = labels.type(torch.float)
        self.data = data.type(torch.float)
        self.length = len(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionnary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
        """
        # Select sample
        X = self.data[index]

        index_bis = torch.randperm(self.length)[index]
        X_bis = self.data[index_bis]
        y = self.labels[index]
        y_bis = self.labels[index_bis]

        return {"data": X, "data_bis": X_bis, "labels": y, "labels_bis": y_bis}
