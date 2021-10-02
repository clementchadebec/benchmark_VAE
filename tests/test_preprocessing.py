import os

import numpy as np
import pytest
import torch

from pyraug.data.loaders import ImageGetterFromFolder
from pyraug.data.preprocessors import DataProcessor

PATH = os.path.dirname(os.path.abspath(__file__))


# data format
class Test_Format_Data:
    @pytest.fixture(params=[[[0.0, "a"]], {"a": 0}])
    def bad_data(self, request):
        return request.param

    @pytest.fixture
    def bad_shape_data(self):
        return np.array([[10.0, 1], [2, 5.0, 0.0], [0.0, 1.0]])

    @pytest.fixture(
        params=[torch.tensor([[np.nan, 0], [12, 0]]), np.array([[np.nan, 0], [12, 0]])]
    )
    def nan_data(self, request):
        return request.param

    def test_raise_bad_data_error(self, bad_data):
        with pytest.raises(TypeError):
            DataProcessor.to_tensor(bad_data)

    @pytest.mark.filterwarnings("ignore")
    def test_raise_bad_shape_error(self, bad_shape_data):
        with pytest.raises(TypeError):
            DataProcessor.to_tensor(bad_shape_data)

    def test_raise_nan_data_error(self, nan_data):
        assert DataProcessor.has_nan(nan_data)


class Test_Apply_Transforms:
    @pytest.fixture(params=["min_max_scaling", "individual_min_max_scaling"])
    def normalization_type(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [
                [np.array([[10.0, 1]]), np.array([[2, 0.0]]), np.array([[0.0, 1.0]])],
                (3, 1, 2),
            ],
            [
                [
                    torch.tensor([[10.0, 1]]),
                    torch.tensor([[2, 0.0]]),
                    torch.tensor([[0.0, 1.0]]),
                ],
                (3, 1, 2),
            ],
            [
                [
                    torch.tensor([[[10.0, 1], [0.0, 1]]]),
                    torch.tensor([[[2, 0.0], [0.0, 1.0]]]),
                ],
                (2, 1, 2, 2),
            ],
            [
                torch.load(
                    os.path.join(PATH, "data/unnormalized_mnist_data_list_of_array")
                ),
                (3, 1, 28, 28),
            ],
            [
                ImageGetterFromFolder.load(
                    os.path.join(PATH, "data/loading/dummy_data_folder")
                ),
                (5, 3, 12, 12),
            ],
        ]
    )
    def unormalized_data(self, request):
        return request.param

    def test_normalize_data(self, unormalized_data, normalization_type):

        with pytest.raises(RuntimeError):
            data_processor = DataProcessor(data_normalization_type=None)
            checked_data = data_processor.process_data(unormalized_data[0])

        data_processor = DataProcessor(data_normalization_type=None)
        checked_data = data_processor.normalize_data(unormalized_data[0])
        # Check nothing happens with normalization set to None
        assert checked_data == unormalized_data[0]

        data_processor = DataProcessor(data_normalization_type=normalization_type)
        checked_data = data_processor.process_data(unormalized_data[0])

        if normalization_type == "min_max_scaling":
            assert bool(all([c_data.min() >= 0 for c_data in checked_data])) and bool(
                all([c_data.max() <= 1 for c_data in checked_data])
            )

        elif normalization_type == "individual_min_max_scaling":
            assert bool(all([c_data.min() == 0 for c_data in checked_data])) and bool(
                all([c_data.max() == 1 for c_data in checked_data])
            )

    def test_data_shape(self, unormalized_data, normalization_type):

        data_processor = DataProcessor(data_normalization_type=normalization_type)
        checked_data = data_processor._process_data_list(unormalized_data[0])
        # assert 0, f"{checked_data}"
        assert checked_data.shape == unormalized_data[1]

    @pytest.fixture(
        params=[
            (torch.randn(1, 10, 30), (15), (1, 10, 15)),
            (torch.randn(1, 3, 30, 30), (20, 20), (1, 3, 20, 20)),
            (torch.randn(1, 5, 20, 20, 20), (10, 15, 9), (1, 5, 10, 15, 9)),
        ]
    )
    def data_to_reshape(self, request):
        return request.param

    def test_reshape_data(self, data_to_reshape):
        data = data_to_reshape[0]
        target_shape = data_to_reshape[1]
        target_output_shape = data_to_reshape[2]

        reshaped_data = DataProcessor._reshape_data(data, target_shape)

        assert reshaped_data.shape == target_output_shape

    @pytest.fixture(
        params=[
            [
                [
                    100 * torch.rand(3, 20, 15, 30),
                    torch.rand(3, 10, 25, 10),
                    torch.rand(3, 10, 10, 30),
                ],
                (3, 3, 10, 10, 10),
            ],
            [
                [
                    100 * torch.rand(1, 20, 15, 30),
                    torch.rand(1, 10, 25, 10),
                    torch.rand(1, 10, 10, 30),
                    10000 * torch.rand(1, 10, 10, 30),
                    100 * torch.rand(1, 100, 30, 30),
                ],
                (5, 1, 10, 10, 10),
            ],
            [torch.randn(4, 12, 10), (4, 12, 10)],
            [np.random.randn(10, 2, 17, 28), (10, 2, 17, 28)],
        ]
    )
    def messy_data(self, request):
        return request.param

    def test_transforms_messy_data(self, messy_data, normalization_type):

        data_processor = DataProcessor(data_normalization_type=normalization_type)
        checked_data = data_processor.process_data(messy_data[0])

        assert checked_data.shape == messy_data[1]
        if normalization_type == "min_max_scaling":
            assert bool(all([c_data.min() >= 0 for c_data in checked_data])) and bool(
                all([c_data.max() <= 1 for c_data in checked_data])
            )
            assert not (
                bool(all([c_data.min() == 0 for c_data in checked_data]))
                and bool(all([c_data.max() == 1 for c_data in checked_data]))
            )

        elif normalization_type == "individual_min_max_scaling":
            assert bool(all([c_data.min() == 0 for c_data in checked_data])) and bool(
                all([c_data.max() == 1 for c_data in checked_data])
            )

    def test_create_dataset(self, messy_data, normalization_type):

        labels = torch.rand(messy_data[1][0])

        data_processor = DataProcessor(data_normalization_type=normalization_type)

        checked_data = data_processor.process_data(messy_data[0])
        dataset = DataProcessor.to_dataset(checked_data, labels)

        assert torch.equal(dataset.data, checked_data)
        assert torch.equal(dataset.labels, labels)
