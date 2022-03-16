import os

import numpy as np
import pytest
import torch
from copy import deepcopy

from pythae.models import VAE, VAEConfig, VAMP, VAMPConfig, AE, AEConfig
from pythae.samplers import TwoStageVAESampler, TwoStageVAESamplerConfig
from pythae.trainers import BaseTrainerConfig
PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def dummy_data():
    ### 3 imgs from mnist that are used to simulated generated ones
    return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample")).data


@pytest.fixture(
    params=[VAMP(VAMPConfig(input_dim=(1, 28, 28))), VAE(VAEConfig(input_dim=(1, 28, 28)))]
)
def model(request):
    return request.param

@pytest.fixture(
    params=[
        TwoStageVAESamplerConfig(
            second_stage_depth=2,
            second_layers_dim=100),
        TwoStageVAESamplerConfig(
            second_stage_depth=0,
            second_layers_dim=1024),
        None,
    ]
)
def sampler_config(request):
    return request.param

@pytest.fixture()
def sampler(model, sampler_config):
    return TwoStageVAESampler(
        model=model, sampler_config=sampler_config
    )


@pytest.fixture(params=[(4, 2), (5, 5), (2, 3)])
def num_sample_and_batch_size(request):
    return request.param

class Test_TwoeStepVAESampler_ModelChecking:

    @pytest.fixture()
    def wrong_model(self):
        return AE(AEConfig(input_dim=(1, 28, 28)))

    def test_raises_wrong_model(self, wrong_model):
        
        with pytest.raises(AssertionError):
            sampler = TwoStageVAESampler(
                model=model)




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
        return BaseTrainerConfig(
            output_dir=dir_path,
            num_epochs=20)

    def test_return_sampling_with_eval(
        self, model, dummy_data, training_config, sampler, num_sample_and_batch_size
    ):

        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        start_gamma = deepcopy(sampler.second_vae.decoder.gamma_z)

        sampler.fit(
            train_data=dummy_data,
            eval_data=dummy_data,
            training_config=training_config)

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
            train_data=dummy_data,
            eval_data=None,
            training_config=training_config)

        final_gamma = deepcopy(sampler.second_vae.decoder.gamma_z)

        gen_samples = sampler.sample(
            num_samples=num_samples, batch_size=batch_size, return_gen=True
        )

        assert gen_samples.shape[0] == num_samples
        assert start_gamma != final_gamma

    def test_save_sampling(
        self, tmpdir, dummy_data, model, training_config, sampler, num_sample_and_batch_size
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
        self, tmpdir, dummy_data, model, training_config, sampler, num_sample_and_batch_size
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


# class Test_Sampler_Set_up:
#    @pytest.fixture(
#        params=[# (target full batch number, target last full batch size, target_batch_number)
#            NormalSamplerConfig(),
#        ]
#    )
#    def sampler_config(self, tmpdir, request):
#        return request.param
#
#    def test_sampler_set_up(self, model, sampler_config):
#        sampler = NormalSampler(model=model, sampler_config=sampler_config)
#
#        assert sampler.batch_size == sampler_config.batch_size
#        assert sampler.samples_per_save == sampler_config.samples_per_save
