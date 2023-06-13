import os

import numpy as np
import pytest
import torch
from imageio import imread

from pythae.models import BaseAE, BaseAEConfig
from pythae.samplers import BaseSampler, BaseSamplerConfig

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def dummy_data():
    ### 3 imgs from mnist that are used to simulated generated ones
    return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample")).data


@pytest.fixture(params=[torch.rand(3, 10, 20), torch.rand(1, 2, 2)])
def img_tensors(request):
    return request.param


@pytest.fixture
def model_sample():
    return BaseAE((BaseAEConfig(input_dim=(1, 28, 28))))


@pytest.fixture()
def sampler_sample(tmpdir, model_sample):
    tmpdir.mkdir("dummy_folder")
    return BaseSampler(model=model_sample, sampler_config=BaseSamplerConfig())


class Test_BaseSampler_saving:
    def test_save_config(self, tmpdir, sampler_sample):
        sampler = sampler_sample

        dir_path = os.path.join(tmpdir, "dummy_folder")

        sampler.save(dir_path)

        sampler_config_file = os.path.join(dir_path, "sampler_config.json")

        assert os.path.isfile(sampler_config_file)

        generation_config_rec = BaseSamplerConfig.from_json_file(sampler_config_file)

        assert generation_config_rec.__dict__ == sampler_sample.sampler_config.__dict__

    def test_save_image_tensor(self, img_tensors, tmpdir, sampler_sample):

        sampler = sampler_sample

        dir_path = os.path.join(tmpdir, "dummy_folder")
        img_path = os.path.join(dir_path, "test_img.png")

        sampler.save_img(img_tensors, dir_path, "test_img.png")

        assert os.path.isdir(dir_path)
        assert os.path.isfile(img_path)

        rec_img = torch.tensor(imread(img_path)) / 255.0

        assert 1 >= rec_img.max() >= 0
