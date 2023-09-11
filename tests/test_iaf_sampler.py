import os
from copy import deepcopy

import pytest
import torch

from pythae.models import AE, VAE, AEConfig, VAEConfig
from pythae.pipelines import GenerationPipeline
from pythae.samplers import (
    IAFSampler,
    IAFSamplerConfig,
    NormalSampler,
    NormalSamplerConfig,
)
from pythae.trainers import BaseTrainerConfig
from pythae.data.preprocessors import DataProcessor

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(
        params=[
            torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample")).data,
            DataProcessor().to_dataset(torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample")).data)]
)
def dummy_data(request):
    ### 3 imgs from mnist that are used to simulated generated ones
    return request.param

@pytest.fixture(
    params=[
        AE(AEConfig(input_dim=(1, 28, 28), latent_dim=2)),
        VAE(VAEConfig(input_dim=(1, 28, 28), latent_dim=4)),
    ]
)
def model(request):
    return request.param


@pytest.fixture(
    params=[
        IAFSamplerConfig(n_made_blocks=2, n_hidden_in_made=1),
        IAFSamplerConfig(hidden_size=12, include_batch_norm=True),
        None,
    ]
)
def sampler_config(request):
    return request.param


@pytest.fixture()
def sampler(model, sampler_config):
    return IAFSampler(model=model, sampler_config=sampler_config)


@pytest.fixture(params=[(4, 2), (5, 5), (2, 3)])
def num_sample_and_batch_size(request):
    return request.param


class Test_IAFSampler_saving:
    def test_save_config(self, tmpdir, sampler):

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")

        sampler.save(dir_path)

        sampler_config_file = os.path.join(dir_path, "sampler_config.json")

        assert os.path.isfile(sampler_config_file)

        generation_config_rec = IAFSamplerConfig.from_json_file(sampler_config_file)

        assert generation_config_rec.__dict__ == sampler.sampler_config.__dict__


class Test_IAFSampler_Sampling:
    @pytest.fixture()
    def training_config(self, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        return BaseTrainerConfig(output_dir=dir_path, num_epochs=20)

    def test_return_sampling_with_eval(
        self, model, dummy_data, training_config, sampler, num_sample_and_batch_size
    ):

        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        start_flow = deepcopy(sampler.flow_contained_model)

        sampler.fit(
            train_data=dummy_data, eval_data=dummy_data, training_config=training_config
        )

        final_flow = deepcopy(sampler.flow_contained_model)

        gen_samples = sampler.sample(
            num_samples=num_samples, batch_size=batch_size, return_gen=True
        )

        assert gen_samples.shape[0] == num_samples
        assert start_flow != final_flow

    def test_return_sampling_without_eval(
        self, dummy_data, training_config, sampler, num_sample_and_batch_size
    ):

        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        start_flow = deepcopy(sampler.flow_contained_model)

        sampler.fit(
            train_data=dummy_data, eval_data=None, training_config=training_config
        )

        final_flow = deepcopy(sampler.flow_contained_model)

        gen_samples = sampler.sample(
            num_samples=num_samples, batch_size=batch_size, return_gen=True
        )

        assert gen_samples.shape[0] == num_samples
        assert start_flow != final_flow

    def test_save_sampling(
        self,
        tmpdir,
        dummy_data,
        training_config,
        sampler,
        num_sample_and_batch_size,
    ):

        dir_path = os.path.join(tmpdir, "dummy_folder")
        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        sampler.fit(train_data=dummy_data, training_config=training_config)

        gen_samples = sampler.sample(
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=dir_path,
            return_gen=True,
        )

        assert gen_samples.shape[0] == num_samples
        assert len(os.listdir(dir_path)) == num_samples

    def test_save_sampling_and_sampler_config(
        self,
        tmpdir,
        dummy_data,
        training_config,
        sampler,
        num_sample_and_batch_size,
    ):

        dir_path = os.path.join(tmpdir, "dummy_folder")
        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        sampler.fit(train_data=dummy_data, training_config=training_config)

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
            assert isinstance(pipe.sampler, IAFSampler)
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
