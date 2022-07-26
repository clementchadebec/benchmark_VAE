import pytest
import os
import torch

from torch.utils.data import Dataset
from pythae.pipelines import *
from pythae.models import VAE, VAEConfig, FactorVAE, FactorVAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.samplers import NormalSampler, NormalSamplerConfig
from pythae.customexception import DatasetError

PATH = os.path.dirname(os.path.abspath(__file__))


class CustomWrongOutputDataset(Dataset):

    def __init__(self, path) -> None:
        self.img_path = path

    def __len__(self):
        return len(torch.load(os.path.join(self.img_path)).data)

    def __getitem__(self, index) -> dict:
        data = torch.load(self.img_path).data[index]
        return {'wrong_key': data}

class CustomNoLenDataset(Dataset):

    def __init__(self, path) -> None:
        self.img_path = path

    def __getitem__(self, index) -> dict:
        data = torch.load(self.img_path).data[index]
        return {'data': data}

class CustomDataset(Dataset):

    def __init__(self, path) -> None:
        self.img_path = path

    def __len__(self):
        return len(torch.load(os.path.join(self.img_path)).data)

    def __getitem__(self, index) -> dict:
        data = torch.load(self.img_path).data[index]
        return {'data': data}


class Test_Pipeline_Standalone:
    @pytest.fixture
    def train_dataset(self):
        return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))

    @pytest.fixture
    def custom_train_dataset(self):
        return CustomDataset(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))

    @pytest.fixture
    def custom_wrong_output_train_dataset(self):
        return CustomWrongOutputDataset(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))

    @pytest.fixture
    def custom_no_len_train_dataset(self):
        return CustomWrongOutputDataset(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))

    @pytest.fixture
    def training_pipeline(self, train_dataset):
        vae_config = VAEConfig(input_dim=tuple(train_dataset.data[0].shape), latent_dim=2)
        vae = VAE(vae_config)
        pipe = TrainingPipeline(model=vae)
        return pipe

    def test_base_pipeline(self):
        with pytest.raises(NotImplementedError):
            pipe = Pipeline()
            pipe()

    def test_training_pipeline(self, tmpdir, training_pipeline, train_dataset):

        with pytest.raises(AssertionError):
            pipeline = TrainingPipeline(
                model=VAE(VAEConfig(input_dim=(1, 2, 3))), training_config=Pipeline()
            )

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        training_pipeline.training_config.output_dir = dir_path
        training_pipeline.training_config.num_epochs = 1
        training_pipeline(train_dataset.data)
        assert isinstance(training_pipeline.model, VAE)


    def test_training_pipeline_wrong_output_dataset(self, tmpdir, training_pipeline, train_dataset, custom_wrong_output_train_dataset):
        
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        training_pipeline.training_config.output_dir = dir_path
        training_pipeline.training_config.num_epochs = 1
        
        with pytest.raises(DatasetError):
            training_pipeline(train_data=custom_wrong_output_train_dataset)
        
        with pytest.raises(DatasetError):
            training_pipeline(train_data=train_dataset.data, eval_data=custom_wrong_output_train_dataset)


    def test_training_pipeline_no_len_dataset(self, tmpdir, training_pipeline, train_dataset, custom_no_len_train_dataset):
        
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        training_pipeline.training_config.output_dir = dir_path
        training_pipeline.training_config.num_epochs = 1
        
        with pytest.raises(DatasetError):
            training_pipeline(train_data=custom_no_len_train_dataset)
        
        with pytest.raises(DatasetError):
            training_pipeline(train_data=train_dataset.data, eval_data=custom_no_len_train_dataset)


    def test_training_pipleine_custom_dataset(self, tmpdir, training_pipeline, train_dataset, custom_train_dataset):


        with pytest.raises(DatasetError):
            vae_config = FactorVAEConfig(input_dim=tuple(train_dataset.data[0].shape), latent_dim=2)
            vae = FactorVAE(vae_config)
            pipe = TrainingPipeline(model=vae)
            pipe(train_data=custom_train_dataset)

        with pytest.raises(DatasetError):
            vae_config = FactorVAEConfig(input_dim=tuple(train_dataset.data[0].shape), latent_dim=2)
            vae = FactorVAE(vae_config)
            pipe = TrainingPipeline(model=vae)
            pipe(train_data=train_dataset.data, eval_data=custom_train_dataset)

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        training_pipeline.training_config.output_dir = dir_path
        training_pipeline.training_config.num_epochs = 1
        training_pipeline(train_data=custom_train_dataset, eval_data=custom_train_dataset)

        assert training_pipeline.trainer.train_dataset == custom_train_dataset
        assert training_pipeline.trainer.eval_dataset == custom_train_dataset

    def test_generation_pipeline(self, tmpdir, train_dataset):

        with pytest.raises(NotImplementedError):
            pipe = GenerationPipeline(model=VAE(VAEConfig(input_dim=(1, 2, 3))), sampler_config=BaseTrainerConfig())
        
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
