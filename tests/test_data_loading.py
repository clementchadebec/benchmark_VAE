import os

import numpy as np
import pytest
import torch

from pythae.customexception import LoadError
from pythae.data.datasets import BaseDataset

PATH = os.path.dirname(os.path.abspath(__file__))


class Test_Dataset:
    @pytest.fixture
    def data(self):
        return torch.tensor([[10.0, 1], [2, 0.0]])

    @pytest.fixture
    def labels(self):
        return torch.tensor([0, 1])

    def test_dataset(self, data, labels):
        dataset = BaseDataset(data, labels)
        assert torch.all(dataset[0]["data"] == data[0])
        assert torch.all(dataset[1]["labels"] == labels[1])
        assert torch.all(dataset.data == data)
        assert torch.all(dataset.labels == labels)
