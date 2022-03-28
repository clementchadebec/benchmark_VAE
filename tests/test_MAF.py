import pytest
import os
import torch
import numpy as np
import shutil

from pythae.models.base.base_utils import ModelOutput
from pythae.models.normalizing_flows import MAF, MAFConfig

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(params=[MAFConfig(n_hidden_in_made=10), MAFConfig(n_made_blocks=2)])
def model_configs_no_input_output_dim(request):
    return request.param


@pytest.fixture(
    params=[
        MAFConfig(input_dim=(1, 8, 2), n_made_blocks=2, n_hidden_in_made=1),
        MAFConfig(input_dim=(1, 2, 18), hidden_size=12),
    ]
)
def model_configs(request):
    return request.param


class Test_Model_Building:
    def test_build_model(self, model_configs):
        model = MAF(model_configs)
        assert all(
            [
                model.input_dim == np.prod(model_configs.input_dim),
            ]
        )

    def test_raises_no_input_output_dim(
        self, model_configs_no_input_output_dim):
        with pytest.raises(AttributeError):
            model = MAF(model_configs_no_input_output_dim)


class Test_Model_Saving:

    def test_creates_saving_path(self, model_configs):
        dir_path = os.path.join(PATH, 'test/for/saving')
        model = MAF(model_configs)
        model.save(dir_path=dir_path)
        shutil.rmtree(dir_path)

        dir_path = None
        model = MAF(model_configs)
        with pytest.raises(TypeError) or pytest.raises(FileNotFoundError):
            model.save(dir_path=dir_path)

    def test_default_model_saving(self, tmpdir, model_configs):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = MAF(model_configs)

        rnd_key = list(model.state_dict().keys())[0]
        model.state_dict()[rnd_key][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(["model_config.json", "model.pt"])

        # reload model
        model_rec = MAF.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_raises_missing_files(
        self, tmpdir, model_configs):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = MAF(model_configs,)

        rnd_key = list(model.state_dict().keys())[0]
        model.state_dict()[rnd_key][0] = 0

        model.save(dir_path=dir_path)

        os.remove(os.path.join(dir_path, "model.pt"))

        # check raises model.pt is missing
        with pytest.raises(FileNotFoundError):
            model_rec = MAF.load_from_folder(dir_path)

        torch.save({"wrong_key": 0.}, os.path.join(dir_path, "model.pt"))
        # check raises wrong key in model.pt
        with pytest.raises(KeyError):
            model_rec = MAF.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "model_config.json"))

        # check raises model_config.json is missing
        with pytest.raises(FileNotFoundError):
            model_rec = MAF.load_from_folder(dir_path)

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
        return MAF(model_configs)

    def test_model_train_output(self, made, demo_data):

        made.train()
        out = made(demo_data['data'])

        assert isinstance(out, ModelOutput)

        assert set(["out", "log_abs_det_jac"]) == set(out.keys())

        assert out.out.shape[0] == demo_data["data"].shape[0]
        assert out.log_abs_det_jac.shape[0] == demo_data["data"].shape[0]
        assert out.out.shape[1:] == np.prod(made.model_config.output_dim)
        assert out.log_abs_det_jac.shape[1:] == np.prod(made.model_config.output_dim)

        out = made.inverse(out.out)

        assert isinstance(out, ModelOutput)

        assert set(["out", "log_abs_det_jac"]) == set(out.keys())

        assert out.out.shape[0] == demo_data["data"].shape[0]
        assert out.log_abs_det_jac.shape[0] == demo_data["data"].shape[0]
        assert out.out.shape[1:] == np.prod(made.model_config.output_dim)
        assert out.log_abs_det_jac.shape[1:] == np.prod(made.model_config.output_dim)
