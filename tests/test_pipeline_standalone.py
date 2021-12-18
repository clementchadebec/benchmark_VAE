import pytest
import os
import torch

from pythae.pipelines import *
from pythae.models import VAE
from pythae.trainers import BaseTrainingConfig

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
        pipe = TrainingPipeline()
        assert isinstance(pipe.training_config, BaseTrainingConfig)
        
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        pipe = TrainingPipeline(
            training_config=BaseTrainingConfig(num_epochs=1, output_dir=dir_path)
        )
        pipe(train_dataset.data)
        assert isinstance(pipe.model, VAE)
        
