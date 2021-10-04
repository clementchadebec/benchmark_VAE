import os

import numpy as np
import pytest
import torch

from pythae.models import AE, AEConfig, VAE, VAEConfig
from pythae.samplers import NormalSampler, NormalSampler_Config

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def dummy_data():
    ### 3 imgs from mnist that are used to simulated generated ones
    return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample")).data


@pytest.fixture(
    params=[
        AE(AEConfig(input_dim=784)),
        VAE(VAEConfig(input_dim=784))
    ]
)
def model(request):
    return request.param


@pytest.fixture()
def sampler_sample(tmpdir, model):
    tmpdir.mkdir("dummy_folder")
    return BaseSampler(
        model=model,
        sampler_config=NormalSamplerConfig(
            batch_size=2
        ),
    )


class Test_NormalSampler_saving:
    def test_save_config(self, tmpdir, sampler_sample):
        sampler = sampler_sample

        dir_path = os.path.join(tmpdir, "dummy_folder")

        sampler.save(dir_path)

        sampler_config_file = os.path.join(dir_path, "sampler_config.json")

        assert os.path.isfile(sampler_config_file)

        generation_config_rec = BaseSamplerConfig.from_json_file(sampler_config_file)

        assert generation_config_rec.__dict__ == sampler_sample.sampler_config.__dict__


class Test_Sampler_Set_up:
    @pytest.fixture(
        params=[
            BaseSamplerConfig(
                batch_size=1
            ),  # (target full batch number, target last full batch size, target_batch_number)
            BaseSamplerConfig(batch_size=2),
        ]
    )
    def sampler_config(self, tmpdir, request):
        return request.param

    def test_sampler_set_up(self, model_sample, sampler_config):
        sampler = BaseSampler(model=model_sample, sampler_config=sampler_config)

        assert sampler.batch_size == sampler_config.batch_size
        assert sampler.samples_per_save == sampler_config.samples_per_save