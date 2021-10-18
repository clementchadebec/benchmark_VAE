import os

import numpy as np
import pytest
import torch

from pythae.models import RHVAE, RHVAEConfig
from pythae.samplers import RHVAESampler, RHVAESamplerConfig

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def dummy_data():
    ### 3 imgs from mnist that are used to simulated generated ones
    return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample")).data


@pytest.fixture(
    params=[
        RHVAE(RHVAEConfig(input_dim=(1, 28, 28), latent_dim=3)),
        RHVAE(RHVAEConfig(input_dim=(1, 28, 28), latent_dim=2)),
    ]
)
def model(request):
    return request.param


@pytest.fixture(
    params=[
        RHVAESamplerConfig(n_lf=1, mcmc_steps_nbr=2, eps_lf=0.00001),
        RHVAESamplerConfig(n_lf=3, mcmc_steps_nbr=2, eps_lf=0.001),
        RHVAESamplerConfig(n_lf=3, mcmc_steps_nbr=2, beta_zero=0.1),
    ]
)
def sampler_config(request):
    return request.param


@pytest.fixture()
def sampler(model, sampler_config):

    # simulates learned metric
    model.centroids_tens = torch.randn(20, model.latent_dim)
    model.M_tens = torch.randn(20, model.latent_dim, model.latent_dim)
    return RHVAESampler(model=model, sampler_config=sampler_config)


@pytest.fixture(params=[(4, 2), (5, 5), (2, 3)])
def num_sample_and_batch_size(request):
    return request.param


class Test_RHVAESampler_saving:
    def test_save_config(self, tmpdir, sampler):

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")

        sampler.save(dir_path)

        sampler_config_file = os.path.join(dir_path, "sampler_config.json")

        assert os.path.isfile(sampler_config_file)

        generation_config_rec = RHVAESamplerConfig.from_json_file(sampler_config_file)

        assert generation_config_rec.__dict__ == sampler.sampler_config.__dict__


class Test_RHVAESampler_Sampling:
    def test_return_sampling(self, model, sampler, num_sample_and_batch_size):

        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        gen_samples = sampler.sample(
            num_samples=num_samples, batch_size=batch_size, return_gen=True
        )

        assert gen_samples.shape[0] == num_samples

    def test_save_sampling(self, tmpdir, model, sampler, num_sample_and_batch_size):

        dir_path = os.path.join(tmpdir, "dummy_folder")
        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

        gen_samples = sampler.sample(
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=dir_path,
            return_gen=True,
        )

        assert gen_samples.shape[0] == num_samples
        assert len(os.listdir(dir_path)) == num_samples

    def test_save_sampling_and_sampler_config(
        self, tmpdir, model, sampler, num_sample_and_batch_size
    ):

        dir_path = os.path.join(tmpdir, "dummy_folder")
        num_samples, batch_size = (
            num_sample_and_batch_size[0],
            num_sample_and_batch_size[1],
        )

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
