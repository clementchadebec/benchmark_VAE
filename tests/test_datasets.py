import os

import pytest
import torch

from pythae.data.datasets import BaseDataset

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def data():
    return torch.randn(1000, 2)


@pytest.fixture
def labels():
    return torch.randint(10, (1000,))


class Test_Dataset:
    def test_dataset_call(self, data, labels):
        dataset = BaseDataset(data, labels)

        assert torch.all(dataset[0]["data"] == data[0])
        assert torch.all(dataset[1]["labels"] == labels[1])
        assert torch.all(dataset.data == data)
        assert torch.all(dataset.labels == labels)

        index = torch.randperm(1000)[:3]

        assert torch.all(dataset[index]["data"] == data[index])
        assert torch.all(dataset[index]["labels"] == labels[index])
