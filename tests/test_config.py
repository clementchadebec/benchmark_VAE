import json
import os

import pytest
from pydantic import ValidationError

from pyraug.config import BaseConfig
from pyraug.models.base.base_config import BaseModelConfig
from pyraug.models.rhvae.rhvae_config import RHVAEConfig, RHVAESamplerConfig
from pyraug.trainers.training_config import TrainingConfig

PATH = os.path.dirname(os.path.abspath(__file__))

# RHVAE loading tests
class Test_Save_Model_JSON_from_Config:
    @pytest.fixture(
        params=[BaseModelConfig(), BaseModelConfig(input_dim=100, latent_dim=5)]
    )
    def model_configs(self, request):
        return request.param

    def test_save_json(self, tmpdir, model_configs):
        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model_configs.save_json(dir_path, "dummy_json")

        assert "dummy_json.json" in os.listdir(dir_path)

        rec_model_config = BaseModelConfig.from_json_file(
            os.path.join(dir_path, "dummy_json.json")
        )

        assert rec_model_config.__dict__ == model_configs.__dict__

    @pytest.fixture(
        params=[TrainingConfig(), TrainingConfig(learning_rate=100, batch_size=15)]
    )
    def training_configs(self, request):
        return request.param

    def test_save_json(self, tmpdir, training_configs):
        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        training_configs.save_json(dir_path, "dummy_json")

        assert "dummy_json.json" in os.listdir(dir_path)

        rec_training_config = TrainingConfig.from_json_file(
            os.path.join(dir_path, "dummy_json.json")
        )

        assert rec_training_config.__dict__ == training_configs.__dict__


class Test_Load_RHVAE_Config_From_JSON:
    @pytest.fixture(
        params=[
            os.path.join(PATH, "data/rhvae/configs/model_config00.json"),
            os.path.join(PATH, "data/rhvae/configs/training_config00.json"),
            os.path.join(PATH, "data/rhvae/configs/generation_config00.json"),
        ]
    )
    def custom_config_path(self, request):
        return request.param

    @pytest.fixture
    def corrupted_config_path(self):
        return "corrupted_path"

    @pytest.fixture
    def not_json_config_path(self):
        return os.path.join(PATH, "data/rhvae/configs/not_json_file.md")

    @pytest.fixture(
        params=[
            [
                os.path.join(PATH, "data/rhvae/configs/model_config00.json"),
                RHVAEConfig(
                    latent_dim=11,
                    n_lf=2,
                    eps_lf=0.00001,
                    temperature=0.5,
                    regularization=0.1,
                    beta_zero=0.8,
                ),
            ],
            [
                os.path.join(PATH, "data/rhvae/configs/training_config00.json"),
                TrainingConfig(
                    batch_size=3,
                    max_epochs=2,
                    learning_rate=1e-5,
                    train_early_stopping=10,
                ),
            ],
            [
                os.path.join(PATH, "data/rhvae/configs/generation_config00.json"),
                RHVAESamplerConfig(
                    batch_size=3, mcmc_steps_nbr=3, n_lf=2, eps_lf=0.003
                ),
            ],
        ]
    )
    def custom_config_path_with_true_config(self, request):
        return request.param

    def test_load_custom_config(self, custom_config_path_with_true_config):

        config_path = custom_config_path_with_true_config[0]
        true_config = custom_config_path_with_true_config[1]

        if config_path == os.path.join(PATH, "data/rhvae/configs/model_config00.json"):
            parsed_config = RHVAEConfig.from_json_file(config_path)

        elif config_path == os.path.join(
            PATH, "data/rhvae/configs/training_config00.json"
        ):
            parsed_config = TrainingConfig.from_json_file(config_path)

        else:
            parsed_config = RHVAESamplerConfig.from_json_file(config_path)

        assert parsed_config == true_config

    def test_load_dict_from_json_config(self, custom_config_path):
        config_dict = BaseConfig._dict_from_json(custom_config_path)
        assert type(config_dict) == dict

    def test_raise_load_file_not_found(self, corrupted_config_path):
        with pytest.raises(FileNotFoundError):
            _ = BaseConfig._dict_from_json(corrupted_config_path)

    def test_raise_not_json_file(self, not_json_config_path):
        with pytest.raises(TypeError):
            _ = BaseConfig._dict_from_json(not_json_config_path)


class Test_Load_Config_From_Dict:
    @pytest.fixture(params=[{"latant_dim": 10}, {"batsh_size": 1}, {"mcmc_steps": 12}])
    def corrupted_keys_dict_config(self, request):
        return request.param

    def test_raise_type_error_corrupted_keys(self, corrupted_keys_dict_config):
        if set(corrupted_keys_dict_config.keys()).issubset(["latant_dim"]):
            with pytest.raises(TypeError):
                RHVAEConfig.from_dict(corrupted_keys_dict_config)

        elif set(corrupted_keys_dict_config.keys()).issubset(["batsh_size"]):
            with pytest.raises(TypeError):
                TrainingConfig.from_dict(corrupted_keys_dict_config)

        else:
            with pytest.raises(TypeError):
                RHVAESamplerConfig.from_dict(corrupted_keys_dict_config)

    @pytest.fixture(
        params=[
            {"latent_dim": "bad_type"},
            {"batch_size": "bad_type"},
            {"mcmc_steps_nbr": "bad_type"},
        ]
    )
    def corrupted_type_dict_config(self, request):
        return request.param

    def test_raise_type_error_corrupted_keys(self, corrupted_type_dict_config):

        if set(corrupted_type_dict_config.keys()).issubset(["latent_dim"]):
            with pytest.raises(ValidationError):
                RHVAEConfig.from_dict(corrupted_type_dict_config)

        elif set(corrupted_type_dict_config.keys()).issubset(["batch_size"]):
            with pytest.raises(ValidationError):
                TrainingConfig.from_dict(corrupted_type_dict_config)

        else:
            with pytest.raises(ValidationError):
                RHVAESamplerConfig.from_dict(corrupted_type_dict_config)
