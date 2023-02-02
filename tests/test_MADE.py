import os

import numpy as np
import pytest
import torch

from pythae.models import AutoModel
from pythae.models.base.base_utils import ModelOutput
from pythae.models.normalizing_flows import MADE, MADEConfig

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(params=[MADEConfig(output_dim=(10, 4)), MADEConfig(input_dim=(5,))])
def model_configs_no_input_output_dim(request):
    return request.param


@pytest.fixture(
    params=[
        MADEConfig(input_dim=(1, 8, 2), output_dim=(1, 10), degrees_ordering="random"),
        MADEConfig(input_dim=(1, 2, 18), output_dim=(1, 5), hidden_sizes=[3, 5, 6]),
    ]
)
def model_configs(request):
    return request.param


class Test_Model_Building:
    def test_build_model(self, model_configs):
        model = MADE(model_configs)
        assert all(
            [
                model.input_dim == np.prod(model_configs.input_dim),
                model.output_dim == np.prod(model_configs.output_dim),
                model.hidden_sizes == model_configs.hidden_sizes,
            ]
        )

    def test_raises_no_input_output_dim(self, model_configs_no_input_output_dim):
        with pytest.raises(AttributeError):
            model = MADE(model_configs_no_input_output_dim)


class Test_Model_Saving:
    def test_creates_saving_path(self, tmpdir, model_configs):
        tmpdir.mkdir("saving")
        dir_path = os.path.join(tmpdir, "saving")
        model = MADE(model_configs)
        model.save(dir_path=dir_path)

        dir_path = None
        model = MADE(model_configs)
        with pytest.raises(TypeError) or pytest.raises(FileNotFoundError):
            model.save(dir_path=dir_path)

    def test_default_model_saving(self, tmpdir, model_configs):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = MADE(model_configs)

        model.state_dict()["net.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "environment.json"]
        )

        # reload model
        model_rec = AutoModel.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_raises_missing_files(self, tmpdir, model_configs):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = MADE(
            model_configs,
        )

        model.state_dict()["net.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        os.remove(os.path.join(dir_path, "model.pt"))

        # check raises model.pt is missing
        with pytest.raises(FileNotFoundError):
            model_rec = AutoModel.load_from_folder(dir_path)

        torch.save({"wrong_key": 0.0}, os.path.join(dir_path, "model.pt"))
        # check raises wrong key in model.pt
        with pytest.raises(KeyError):
            model_rec = AutoModel.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "model_config.json"))

        # check raises model_config.json is missing
        with pytest.raises(FileNotFoundError):
            model_rec = AutoModel.load_from_folder(dir_path)


class Test_Model_forward:
    @pytest.fixture
    def demo_data(self):
        data = torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))[
            :
        ]
        return data  # This is an extract of 3 data from MNIST (unnormalized) used to test custom architecture

    @pytest.fixture
    def made(self, model_configs, demo_data):
        model_configs.input_dim = tuple(demo_data["data"][0].shape)
        model_configs.output_dim = tuple(demo_data["data"][0].shape)
        return MADE(model_configs)

    def test_model_train_output(self, made, demo_data):

        made.train()
        out = made(demo_data["data"])

        assert isinstance(out, ModelOutput)

        assert set(["mu", "log_var"]) == set(out.keys())

        assert out.mu.shape[0] == demo_data["data"].shape[0]
        assert out.log_var.shape[0] == demo_data["data"].shape[0]
        assert out.mu.shape[1:] == np.prod(made.model_config.output_dim)
        assert out.log_var.shape[1:] == np.prod(made.model_config.output_dim)
