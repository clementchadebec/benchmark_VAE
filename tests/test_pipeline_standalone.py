import os

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from pythae.customexception import DatasetError
from pythae.data.datasets import DatasetOutput
from pythae.models import VAE, VAEConfig, Adversarial_AE, Adversarial_AE_Config, RAE_L2, RAE_L2_Config, VAEGAN, VAEGANConfig
from pythae.pipelines import *
from pythae.samplers import NormalSampler, NormalSamplerConfig
from pythae.trainers import BaseTrainerConfig

PATH = os.path.dirname(os.path.abspath(__file__))


class CustomWrongOutputDataset(Dataset):
    def __init__(self, path) -> None:
        self.img_path = path

    def __len__(self):
        return len(torch.load(os.path.join(self.img_path)).data)

    def __getitem__(self, index) -> dict:
        data = torch.load(self.img_path).data[index]
        return DatasetOutput(wrong_key=data)


class CustomNoLenDataset(Dataset):
    def __init__(self, path) -> None:
        self.img_path = path

    def __getitem__(self, index) -> dict:
        data = torch.load(self.img_path).data[index]
        return DatasetOutput(data=data)


class CustomDataset(Dataset):
    def __init__(self, path) -> None:
        self.img_path = path

    def __len__(self):
        return len(torch.load(os.path.join(self.img_path)).data)

    def __getitem__(self, index) -> dict:
        data = torch.load(self.img_path).data[index]
        return DatasetOutput(data=data)


class Test_Pipeline_Standalone:
    @pytest.fixture
    def train_dataset(self):
        return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))

    @pytest.fixture
    def custom_train_dataset(self):
        return CustomDataset(
            os.path.join(PATH, "data/mnist_clean_train_dataset_sample")
        )

    @pytest.fixture
    def custom_wrong_output_train_dataset(self):
        return CustomWrongOutputDataset(
            os.path.join(PATH, "data/mnist_clean_train_dataset_sample")
        )

    @pytest.fixture
    def custom_no_len_train_dataset(self):
        return CustomWrongOutputDataset(
            os.path.join(PATH, "data/mnist_clean_train_dataset_sample")
        )
    
    @pytest.fixture(
            params=[
                (VAE, VAEConfig),
                (Adversarial_AE, Adversarial_AE_Config),
                (RAE_L2, RAE_L2_Config),
                (VAEGAN, VAEGANConfig)
            ]
    )
    def model(self, request, train_dataset):
        model = request.param[0](request.param[1](input_dim=tuple(train_dataset.data[0].shape), latent_dim=2))
        
        return model

    @pytest.fixture
    def train_dataloader(self, custom_train_dataset):
        return DataLoader(dataset=custom_train_dataset, batch_size=32)

    @pytest.fixture
    def training_pipeline(self, model, train_dataset):
        vae_config = VAEConfig(
            input_dim=tuple(train_dataset.data[0].shape), latent_dim=2
        )
        vae = VAE(vae_config)
        pipe = TrainingPipeline(model=model)
        return pipe

    def test_base_pipeline(self):
        with pytest.raises(NotImplementedError):
            pipe = Pipeline()
            pipe()

    def test_training_pipeline(self, tmpdir, training_pipeline, train_dataset, model):

        with pytest.raises(AssertionError):
            pipeline = TrainingPipeline(
                model=VAE(VAEConfig(input_dim=(1, 2, 3))), training_config=Pipeline()
            )

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        training_pipeline.training_config.output_dir = dir_path
        training_pipeline.training_config.num_epochs = 1
        training_pipeline(train_dataset.data)
        assert isinstance(training_pipeline.model, model.__class__)

        if model.__class__ == RAE_L2:
            assert training_pipeline.trainer.decoder_optimizer.state_dict()['param_groups'][0]['weight_decay'] == model.model_config.reg_weight

    def test_training_pipeline_wrong_output_dataset(
        self,
        tmpdir,
        training_pipeline,
        train_dataset,
        custom_wrong_output_train_dataset,
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        training_pipeline.training_config.output_dir = dir_path
        training_pipeline.training_config.num_epochs = 1

        with pytest.raises(DatasetError):
            training_pipeline(train_data=custom_wrong_output_train_dataset)

        with pytest.raises(DatasetError):
            training_pipeline(
                train_data=train_dataset.data,
                eval_data=custom_wrong_output_train_dataset,
            )

    def test_training_pipeline_no_len_dataset(
        self, tmpdir, training_pipeline, train_dataset, custom_no_len_train_dataset
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        training_pipeline.training_config.output_dir = dir_path
        training_pipeline.training_config.num_epochs = 1

        with pytest.raises(DatasetError):
            training_pipeline(train_data=custom_no_len_train_dataset)

        with pytest.raises(DatasetError):
            training_pipeline(
                train_data=train_dataset.data, eval_data=custom_no_len_train_dataset
            )

    def test_training_pipleine_custom_dataset(
        self, tmpdir, training_pipeline, train_dataset, custom_train_dataset
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        training_pipeline.training_config.output_dir = dir_path
        training_pipeline.training_config.num_epochs = 1
        training_pipeline(
            train_data=custom_train_dataset, eval_data=custom_train_dataset
        )

        assert training_pipeline.trainer.train_dataset == custom_train_dataset
        assert training_pipeline.trainer.eval_dataset == custom_train_dataset

    def test_generation_pipeline(self, tmpdir, train_dataset):

        with pytest.raises(NotImplementedError):
            pipe = GenerationPipeline(
                model=VAE(VAEConfig(input_dim=(1, 2, 3))),
                sampler_config=BaseTrainerConfig(),
            )

        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        pipe = GenerationPipeline(model=VAE(VAEConfig(input_dim=(1, 2, 3))))
        assert isinstance(pipe.sampler, NormalSampler)
        assert pipe.sampler.sampler_config == NormalSamplerConfig()

        gen_data = pipe(
            num_samples=1,
            batch_size=10,
            output_dir=dir_path,
            return_gen=True,
            save_sampler_config=True,
            train_data=train_dataset.data,
            eval_data=None,
        )

        assert tuple(gen_data.shape) == (1,) + (1, 2, 3)
        assert len(os.listdir(dir_path)) == 1 + 1
        assert "sampler_config.json" in os.listdir(dir_path)

    def test_training_pipeline_with_dataloader(
        self, tmpdir, training_pipeline, train_dataloader
    ):
        # Simulate a training run with a DataLoader
        tmpdir.mkdir("dataloader_test")
        dir_path = os.path.join(tmpdir, "dataloader_test")
        training_pipeline.training_config.output_dir = dir_path
        training_pipeline.training_config.num_epochs = 1
        training_pipeline(train_data=train_dataloader, eval_data=train_dataloader)

        assert isinstance(training_pipeline.trainer.train_loader, DataLoader)
        assert isinstance(training_pipeline.trainer.eval_loader, DataLoader)
