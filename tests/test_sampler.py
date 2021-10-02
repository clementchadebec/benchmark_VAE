import os

import numpy as np
import pytest
import torch

from pyraug.models import RHVAE, BaseVAE
from pyraug.models.base.base_config import BaseModelConfig, BaseSamplerConfig
from pyraug.models.base.base_sampler import BaseSampler
from pyraug.models.rhvae import RHVAEConfig, RHVAESamplerConfig
from pyraug.models.rhvae.rhvae_sampler import RHVAESampler

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def dummy_data():
    ### 3 imgs from mnist that are used to simulated generated ones
    return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample")).data


@pytest.fixture
def model_sample():
    return BaseVAE((BaseModelConfig(input_dim=784)))


@pytest.fixture()
def sampler_sample(tmpdir, model_sample):
    tmpdir.mkdir("dummy_folder")
    return BaseSampler(
        model=model_sample,
        sampler_config=BaseSamplerConfig(
            output_dir=os.path.join(tmpdir, "dummy_folder"), batch_size=2
        ),
    )


class Test_data_saving:
    def test_save_data(self, tmpdir, sampler_sample, dummy_data):

        sampler = sampler_sample

        dir_path = os.path.join(tmpdir, "dummy_folder")

        sampler_sample.save_data_batch(
            dummy_data, dir_path, number_of_samples=3, batch_idx=0
        )

        batch_file = os.path.join(dir_path, "generated_data_3_0.pt")

        assert os.path.isfile(batch_file)

        data_rec = torch.load(batch_file)

        assert torch.equal(dummy_data, data_rec)

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
        tmpdir.mkdir("dummy_folder")
        request.param.output_dir = os.path.join(tmpdir, "dummy_folder")
        return request.param

    def test_sampler_set_up(self, model_sample, sampler_config):
        sampler = BaseSampler(model=model_sample, sampler_config=sampler_config)

        assert sampler.batch_size == sampler_config.batch_size
        assert sampler.samples_per_save == sampler_config.samples_per_save


class Test_RHVAE_Sampler:
    @pytest.fixture(
        params=[
            RHVAESamplerConfig(batch_size=1, mcmc_steps_nbr=15, samples_per_save=5),
            RHVAESamplerConfig(batch_size=2, mcmc_steps_nbr=15, samples_per_save=1),
            RHVAESamplerConfig(
                batch_size=3, n_lf=1, eps_lf=0.01, mcmc_steps_nbr=10, samples_per_save=5
            ),
            RHVAESamplerConfig(
                batch_size=3, n_lf=1, eps_lf=0.01, mcmc_steps_nbr=10, samples_per_save=3
            ),
            RHVAESamplerConfig(
                batch_size=10,
                n_lf=1,
                eps_lf=0.01,
                mcmc_steps_nbr=10,
                samples_per_save=3,
            ),
        ]
    )
    def rhvae_sampler_config(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        request.param.output_dir = os.path.join(tmpdir, "dummy_folder")
        return request.param

    @pytest.fixture(
        params=[
            np.random.randint(1, 15),
            np.random.randint(1, 15),
            np.random.randint(1, 15),
        ]
    )
    def samples_number(self, request):
        return request.param

    @pytest.fixture(
        params=[
            RHVAE(RHVAEConfig(input_dim=784, latent_dim=2)),
            RHVAE(RHVAEConfig(input_dim=784, latent_dim=3)),
        ]
    )
    def rhvae_sample(self, request):
        return request.param

    def test_hmc_sampling(self, rhvae_sample, rhvae_sampler_config):

        # simulates a trained model
        # rhvae_sample.centroids_tens = torch.randn(20, rhvae_sample.latent_dim)
        # rhvae_sample.M_tens = torch.randn(20, rhvae_sample.latent_dim, rhvae_sample.latent_dim)

        sampler = RHVAESampler(model=rhvae_sample, sampler_config=rhvae_sampler_config)

        out = sampler.hmc_sampling(rhvae_sampler_config.batch_size)

        assert out.shape == (rhvae_sampler_config.batch_size, rhvae_sample.latent_dim)

        assert sampler.eps_lf == rhvae_sampler_config.eps_lf

        assert all(
            [
                not torch.equal(out[i], out[j])
                for i in range(len(out))
                for j in range(i + 1, len(out))
            ]
        )

    def test_sampling_loop_saving(
        self, tmpdir, rhvae_sample, rhvae_sampler_config, samples_number
    ):

        sampler = RHVAESampler(model=rhvae_sample, sampler_config=rhvae_sampler_config)
        sampler.sample(samples_number=samples_number)

        generation_folder = os.path.join(tmpdir, "dummy_folder")
        generation_folder_list = os.listdir(generation_folder)

        assert f"generation_{sampler._sampling_signature}" in generation_folder_list

        data_folder = os.path.join(
            generation_folder, f"generation_{sampler._sampling_signature}"
        )
        files_list = os.listdir(data_folder)

        full_data_file_nbr = int(samples_number / rhvae_sampler_config.samples_per_save)
        last_file_data_nbr = samples_number % rhvae_sampler_config.samples_per_save

        if last_file_data_nbr == 0:
            expected_num_of_data_files = full_data_file_nbr
        else:
            expected_num_of_data_files = full_data_file_nbr + 1

        assert len(files_list) == 1 + expected_num_of_data_files

        assert "sampler_config.json" in files_list

        assert all(
            [
                f"generated_data_{rhvae_sampler_config.samples_per_save}_{i}.pt"
                in files_list
                for i in range(full_data_file_nbr)
            ]
        )

        if last_file_data_nbr > 0:
            assert (
                f"generated_data_{last_file_data_nbr}_{expected_num_of_data_files-1}.pt"
                in files_list
            )

        data_rec = []

        for i in range(full_data_file_nbr):
            data_rec.append(
                torch.load(
                    os.path.join(
                        data_folder,
                        "generated_data_"
                        f"{rhvae_sampler_config.samples_per_save}_{i}.pt",
                    )
                )
            )

        if last_file_data_nbr > 0:
            data_rec.append(
                torch.load(
                    os.path.join(
                        data_folder,
                        f"generated_data_"
                        f"{last_file_data_nbr}_{expected_num_of_data_files-1}.pt",
                    )
                )
            )

        data_rec = torch.cat(data_rec)
        assert data_rec.shape[0] == samples_number

        # check sampler_config

        sampler_config_rec = RHVAESamplerConfig.from_json_file(
            os.path.join(data_folder, "sampler_config.json")
        )

        assert sampler_config_rec.__dict__ == rhvae_sampler_config.__dict__
