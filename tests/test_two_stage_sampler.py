import os
from copy import deepcopy

import numpy as np
import pytest
import torch

from pythae.models import AE, VAE, VAMP, AEConfig, VAEConfig, VAMPConfig
from pythae.pipelines import GenerationPipeline
from pythae.samplers import (
    NormalSampler,
    NormalSamplerConfig,
    TwoStageVAESampler,
    TwoStageVAESamplerConfig,
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
        VAMP(VAMPConfig(input_dim=(1, 28, 28), latent_dim=2)),
        VAE(VAEConfig(input_dim=(1, 28, 28), latent_dim=4)),
    ]
)
def model(request):
    return request.param


@pytest.fixture(
    params=[
        TwoStageVAESamplerConfig(second_stage_depth=2, second_layers_dim=100),
        TwoStageVAESamplerConfig(second_stage_depth=0, second_layers_dim=1024),
        None,
    ]
)
def sampler_config(request):
    return request.param


@pytest.fixture()
def sampler(model, sampler_config):
    return TwoStageVAESampler(model=model, sampler_config=sampler_config)


@pytest.fixture(params=[(4, 2), (5, 5), (2, 3)])
def num_sample_and_batch_size(request):
    return request.param


class Test_TwoeStepVAESampler_ModelChecking:
    @pytest.fixture()
    def wrong_model(self):
        return AE(AEConfig(input_dim=(1, 28, 28)))

    def test_raises_wrong_model(self, wrong_model):

        with pytest.raises(AssertionError):
            sampler = TwoStageVAESampler(model=model)


class Test_TwoStageVAESampler_saving:
    def test_save_config(self, tmpdir, sampler):

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")

        sampler.save(dir_path)

        sampler_config_file = os.path.join(dir_path, "sampler_config.json")

        assert os.path.isfile(sampler_config_file)

        generation_config_rec = TwoStageVAESamplerConfig.from_json_file(
            sampler_config_file
        )

        assert generation_config_rec.__dict__ == sampler.sampler_config.__dict__


class Test_TwoStageVAESampler_Sampling:
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

        start_gamma = deepcopy(sampler.second_vae.decoder.gamma_z)

        sampler.fit(
            train_data=dummy_data, eval_data=dummy_data, training_config=training_config
        )

        final_gamma = deepcopy(sampler.second_vae.decoder.gamma_z)

        gen_samples = sampler.sample(
            num_samples=num_samples, batch_size=batch_size, return_gen=True
        )

        assert gen_samples.shape[0] == num_samples
        assert start_gamma != final_gamma

    def test_return_sampling_without_eval(
        self, model, dummy_data, training_config, sampler, num_sample_and_batch_size
    ):

        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        start_gamma = deepcopy(sampler.second_vae.decoder.gamma_z)

        sampler.fit(
            train_data=dummy_data, eval_data=None, training_config=training_config
        )

        final_gamma = deepcopy(sampler.second_vae.decoder.gamma_z)

        gen_samples = sampler.sample(
            num_samples=num_samples, batch_size=batch_size, return_gen=True
        )

        assert gen_samples.shape[0] == num_samples
        assert start_gamma != final_gamma

    def test_save_sampling(
        self,
        tmpdir,
        dummy_data,
        model,
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
        model,
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
            assert isinstance(pipe.sampler, TwoStageVAESampler)
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
