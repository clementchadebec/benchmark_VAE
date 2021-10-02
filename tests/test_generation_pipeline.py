import os

import numpy as np
import pytest
import torch

from pyraug.models import RHVAE
from pyraug.models.rhvae import RHVAEConfig, RHVAESamplerConfig
from pyraug.models.rhvae.rhvae_sampler import RHVAESampler
from pyraug.pipelines import GenerationPipeline


class Test_Pipeline:
    @pytest.fixture(
        params=[
            RHVAEConfig(input_dim=10),
            RHVAEConfig(input_dim=129, latent_dim=2),
            RHVAEConfig(input_dim=1, latent_dim=3, n_lf=2, eps_lf=6),
        ]
    )
    def rhvae_config(self, request):
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
        # request.param.output_dir = os.path.join(tmpdir, "dummy_folder")
        return request.param

    def test_pipeline(self, tmpdir, rhvae_config, rhvae_sampler_config, samples_number):

        tmpdir.mkdir("dummy_folder")
        rhvae_sampler_config.output_dir = os.path.join(tmpdir, "dummy_folder")

        model = RHVAE(rhvae_config)
        sampler = RHVAESampler(model, rhvae_sampler_config)
        pipe = GenerationPipeline(model=model, sampler=sampler)

        # launch pipeline
        pipe(samples_number=samples_number)

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
