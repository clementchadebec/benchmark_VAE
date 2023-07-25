import os

import numpy as np
import pytest
import torch

from pythae.models import PoincareVAE, PoincareVAEConfig
from pythae.pipelines import GenerationPipeline
from pythae.samplers import (
    NormalSampler,
    NormalSamplerConfig,
    PoincareDiskSampler,
    PoincareDiskSamplerConfig,
)

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def dummy_data():
    ### 3 imgs from mnist that are used to simulated generated ones
    return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample")).data


@pytest.fixture(
    params=[
        PoincareVAE(
            PoincareVAEConfig(
                input_dim=(1, 28, 28),
                latent_dim=7,
                prior_distribution="wrapped_normal",
                curvature=0.2,
            )
        ),
        PoincareVAE(
            PoincareVAEConfig(
                input_dim=(1, 28, 28),
                latent_dim=2,
                prior_distribution="riemannian_normal",
                curvature=0.7,
            )
        ),
    ]
)
def model(request):
    return request.param


@pytest.fixture(
    params=[
        PoincareDiskSamplerConfig(),
        None,
    ]
)
def sampler_config(request):
    return request.param


@pytest.fixture()
def sampler(model, sampler_config):
    return PoincareDiskSampler(model=model, sampler_config=sampler_config)


@pytest.fixture(params=[(4, 2), (5, 5), (2, 3)])
def num_sample_and_batch_size(request):
    return request.param


class Test_PoicareSampler_saving:
    def test_save_config(self, tmpdir, sampler):

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")

        sampler.save(dir_path)

        sampler_config_file = os.path.join(dir_path, "sampler_config.json")

        assert os.path.isfile(sampler_config_file)

        generation_config_rec = PoincareDiskSamplerConfig.from_json_file(
            sampler_config_file
        )

        assert generation_config_rec.__dict__ == sampler.sampler_config.__dict__


class Test_PoicareSampler_Sampling:
    def test_return_sampling(
        self, model, dummy_data, sampler, num_sample_and_batch_size
    ):

        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        sampler.fit(train_data=dummy_data)

        gen_samples = sampler.sample(
            num_samples=num_samples, batch_size=batch_size, return_gen=True
        )

        assert gen_samples.shape[0] == num_samples

    def test_save_sampling(
        self, tmpdir, dummy_data, model, sampler, num_sample_and_batch_size
    ):

        dir_path = os.path.join(tmpdir, "dummy_folder")
        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        sampler.fit(train_data=dummy_data)

        gen_samples = sampler.sample(
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=dir_path,
            return_gen=True,
        )

        assert gen_samples.shape[0] == num_samples
        assert len(os.listdir(dir_path)) == num_samples

    def test_save_sampling_and_sampler_config(
        self, tmpdir, dummy_data, model, sampler, num_sample_and_batch_size
    ):

        dir_path = os.path.join(tmpdir, "dummy_folder")
        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        sampler.fit(train_data=dummy_data)

        gen_samples = sampler.sample(
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=dir_path,
            return_gen=True,
            save_sampler_config=True,
        )

        assert gen_samples.shape[0] == num_samples
        assert len(os.listdir(dir_path)) == num_samples + 1
        assert "sampler_config.json" in os.listdir(dir_path)

    def test_generation_pipeline(
        self, tmpdir, dummy_data, model, sampler_config, num_sample_and_batch_size
    ):

        dir_path = os.path.join(tmpdir, "dummy_folder1")
        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        pipe = GenerationPipeline(model=model, sampler_config=None)

        assert isinstance(pipe.sampler, NormalSampler)
        assert pipe.sampler.sampler_config == NormalSamplerConfig()

        gen_data = pipe(
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=dir_path,
            return_gen=True,
            save_sampler_config=True,
            train_data=dummy_data,
            eval_data=None,
        )

        assert tuple(gen_data.shape) == (num_samples,) + tuple(
            model.model_config.input_dim
        )
        assert len(os.listdir(dir_path)) == num_samples + 1
        assert "sampler_config.json" in os.listdir(dir_path)

        dir_path = os.path.join(tmpdir, "dummy_folder2")

        pipe = GenerationPipeline(model=model, sampler_config=sampler_config)

        if sampler_config is None:
            assert isinstance(pipe.sampler, NormalSampler)

        else:
            assert isinstance(pipe.sampler, PoincareDiskSampler)
            assert pipe.sampler.sampler_config == sampler_config

        gen_data = pipe(
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=dir_path,
            return_gen=False,
            save_sampler_config=False,
            train_data=dummy_data,
            eval_data=dummy_data,
        )

        assert gen_data is None
        assert "sampler_config.json" not in os.listdir(dir_path)
        assert len(os.listdir(dir_path)) == num_samples
