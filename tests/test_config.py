import json
import os

import pytest
from pydantic import ValidationError

from pythae.config import BaseConfig
from pythae.models import AEConfig, BaseAEConfig
from pythae.samplers import BaseSamplerConfig, NormalSamplerConfig
from pythae.trainers import AdversarialTrainerConfig, BaseTrainerConfig

PATH = os.path.dirname(os.path.abspath(__file__))


class Test_Load_Config_from_JSON:
    @pytest.fixture(
        params=[
            os.path.join(PATH, "data/baseAE/configs/model_config00.json"),
            os.path.join(PATH, "data/baseAE/configs/training_config00.json"),
            os.path.join(PATH, "data/baseAE/configs/generation_config00.json"),
        ]
    )
    def custom_config_path(self, request):
        return request.param

    @pytest.fixture
    def corrupted_config_path(self):
        return "corrupted_path"

    @pytest.fixture(
        params=[
            [
                os.path.join(PATH, "data/baseAE/configs/model_config00.json"),
                BaseAEConfig(latent_dim=11),
            ],
            [
                os.path.join(PATH, "data/baseAE/configs/training_config00.json"),
                BaseTrainerConfig(
                    per_device_train_batch_size=13,
                    per_device_eval_batch_size=42,
                    num_epochs=2,
                    learning_rate=1e-5,
                ),
            ],
            [
                os.path.join(PATH, "data/baseAE/configs/generation_config00.json"),
                BaseSamplerConfig(),
            ],
        ]
    )
    def custom_config_path_with_true_config(self, request):
        return request.param

    @pytest.fixture
    def not_json_config_path(self):
        return os.path.join(PATH, "data/baseAE/configs/not_json_file.md")

    def test_load_custom_config(self, custom_config_path_with_true_config):

        config_path = custom_config_path_with_true_config[0]
        true_config = custom_config_path_with_true_config[1]

        if config_path == os.path.join(PATH, "data/baseAE/configs/model_config00.json"):
            parsed_config = BaseAEConfig.from_json_file(config_path)

        elif config_path == os.path.join(
            PATH, "data/baseAE/configs/training_config00.json"
        ):
            parsed_config = BaseTrainerConfig.from_json_file(config_path)

        else:
            parsed_config = BaseSamplerConfig.from_json_file(config_path)

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

    def test_raises_user_warning(self, custom_config_path_with_true_config):
        config_path = custom_config_path_with_true_config[0]
        with pytest.warns(UserWarning):
            if config_path == os.path.join(
                PATH, "data/baseAE/configs/model_config00.json"
            ):
                parsed_config = AEConfig.from_json_file(config_path)

            elif config_path == os.path.join(
                PATH, "data/baseAE/configs/training_config00.json"
            ):
                parsed_config = AdversarialTrainerConfig.from_json_file(config_path)

            else:
                parsed_config = NormalSamplerConfig.from_json_file(config_path)


class Test_Save_Model_JSON_from_Config:
    @pytest.fixture(
        params=[BaseAEConfig(), BaseAEConfig(input_dim=(2, 3, 100), latent_dim=5)]
    )
    def model_configs(self, request):
        return request.param

    def test_save_json(self, tmpdir, model_configs):
        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model_configs.save_json(dir_path, "dummy_json")

        assert "dummy_json.json" in os.listdir(dir_path)

        rec_model_config = BaseAEConfig.from_json_file(
            os.path.join(dir_path, "dummy_json.json")
        )

        assert rec_model_config.__dict__ == model_configs.__dict__

    @pytest.fixture(
        params=[
            BaseTrainerConfig(),
            BaseTrainerConfig(
                learning_rate=100,
                per_device_train_batch_size=15,
                per_device_eval_batch_size=23,
            ),
        ]
    )
    def training_configs(self, request):
        return request.param

    def test_save_json(self, tmpdir, training_configs):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")

        training_configs.save_json(dir_path, "dummy_json")

        assert "dummy_json.json" in os.listdir(dir_path)

        rec_training_config = BaseTrainerConfig.from_json_file(
            os.path.join(dir_path, "dummy_json.json")
        )

        assert rec_training_config.__dict__ == training_configs.__dict__
