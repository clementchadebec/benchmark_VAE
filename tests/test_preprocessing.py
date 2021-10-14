import os

import numpy as np
import pytest
import torch

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
