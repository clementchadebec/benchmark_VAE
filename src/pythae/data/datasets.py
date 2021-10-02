"""The pyraug's Datasets inherit from
:class:`torch.utils.data.Dataset` and must be used to convert the data before
training. As of today, it only contains the :class:`pyraug.data.BaseDatset` useful to train a
VAE model but other Datatsets will be added as models are added.
"""
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """This class is the Base class for pyraug's dataset

    A ``__getitem__`` is redefined and outputs a python dictionnary
    with the keys corresponding to `data`, `labels` etc...
    This Class should be used for any new data sets.
    """

    def __init__(self, digits, labels, binarize=False):

        self.labels = labels.type(torch.float)

        if binarize:
            self.data = (torch.rand_like(digits) < digits).type(torch.float)

        else:
            self.data = digits.type(torch.float)

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

        # Load data and get label
        # X = torch.load('data/' + DATA + '.pt')
        y = self.labels[index]

        return {"data": X, "labels": y}
