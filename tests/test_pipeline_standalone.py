import pytest
import os
import torch

from pythae.pipelines import *
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.samplers import NormalSampler, NormalSamplerConfig

PATH = os.path.dirname(os.path.abspath(__file__))


class Test_Pipeline_Standalone:
    @pytest.fixture
    def train_dataset(self):
        return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))

    def test_base_pipeline(self):
        with pytest.raises(NotImplementedError):
            pipe = Pipeline()
            pipe()

    def test_training_pipeline(self, tmpdir, train_dataset):

        with pytest.raises(AssertionError):
            pipeline = TrainingPipeline(
                model=VAE(VAEConfig(input_dim=(1, 2, 3))), training_config=Pipeline()
            )

        pipe = TrainingPipeline()
        assert isinstance(pipe.training_config, BaseTrainerConfig)

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        pipe = TrainingPipeline()
        pipe.training_config.output_dir = dir_path
        pipe.training_config.num_epochs = 1
        pipe(train_dataset.data)
        assert isinstance(pipe.model, VAE)

    def test_generation_pipeline(self, tmpdir, train_dataset):
        
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        pipe = GenerationPipeline(model=VAE(VAEConfig(input_dim=(1, 2, 3))))
        assert isinstance(pipe.sampler, NormalSampler)
        assert pipe.sampler.sampler_config == NormalSamplerConfig()

        gen_data = pipe(num_samples=1,
            batch_size=10,
            output_dir=dir_path,
            return_gen=True,
            save_sampler_config=True,
            train_data=train_dataset.data,
            eval_data=None
        )

        assert tuple(gen_data.shape) == (1,) + (1, 2, 3)
        assert len(os.listdir(dir_path)) == 1 + 1
        assert "sampler_config.json" in os.listdir(dir_path)

