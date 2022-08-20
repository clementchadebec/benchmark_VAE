"""The pythae's Datasets inherit from
:class:`torch.utils.data.Dataset` and must be used to convert the data before
training. As of today, it only contains the :class:`pythae.data.BaseDatset` useful to train a
VAE model but other Datatsets will be added as models are added.
"""
import torch
from torch.utils.data import Dataset


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

        return {"data": X, "labels": y}


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
