import os

import numpy as np
import pytest
import torch

from pythae.data.datasets import BaseDataset
from pythae.data.preprocessors import DataProcessor

PATH = os.path.dirname(os.path.abspath(__file__))


# data format
class Test_Format_Data:
    @pytest.fixture(params=[[[0.0, "a"]], {"a": 0}])
    def bad_data(self, request):
        return request.param

    @pytest.fixture(
        params=[torch.tensor([[np.nan, 0], [12, 0]]), np.array([[np.nan, 0], [12, 0]])]
    )
    def nan_data(self, request):
        return request.param

    def test_raise_bad_data_error(self, bad_data):
        with pytest.raises(TypeError):
            DataProcessor.to_tensor(bad_data)

    def test_raise_nan_data_error(self, nan_data):
        assert DataProcessor.has_nan(nan_data)


class Test_Data_Convert:
    @pytest.fixture(
        params=[
            torch.randn(50, 2, 3),
            np.random.randn(50, 12, 3, 6),
            torch.randn(60, 2, 3),
            np.random.randn(53, 20, 3, 0),
        ]
    )
    def data_to_process(self, request):
        return request.param

    def test_process_data(self, data_to_process):
        data_processor = DataProcessor()
        train_data = data_processor.process_data(data_to_process, batch_size=50)

        if torch.is_tensor(data_to_process):
            assert torch.equal(train_data, data_to_process)

        else:
            assert torch.equal(
                train_data, torch.tensor(data_to_process).type(torch.float)
            ), train_data.shape

        train_dataset = data_processor.to_dataset(train_data)
        assert torch.equal(train_dataset.data, train_data)

    def test_dataset_instance(self, data_to_process):
        data_processor = DataProcessor()
        train_data = data_processor.process_data(data_to_process, batch_size=50)

        train_dataset = data_processor.to_dataset(train_data)
        assert isinstance(train_dataset, BaseDataset)
