import os
from copy import deepcopy

import pytest
import torch
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop

from pythae.customexception import ModelError
from pythae.models import AE, AEConfig, RHVAE, RHVAEConfig
from pythae.trainers import CoupledOptimizerTrainer, CoupledOptimizerTrainerConfig
from tests.data.custom_architectures import *

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def train_dataset():
    return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))


@pytest.fixture()
def model_sample():
    return AE(AEConfig(input_dim=(1, 28, 28)))


@pytest.fixture
def training_config(tmpdir):
    tmpdir.mkdir("dummy_folder")
    dir_path = os.path.join(tmpdir, "dummy_folder")
    return CoupledOptimizerTrainerConfig(output_dir=dir_path)


class Test_DataLoader:
    @pytest.fixture(
        params=[
            CoupledOptimizerTrainerConfig(decoder_optim_decay=0),
            CoupledOptimizerTrainerConfig(batch_size=100, encoder_optim_decay=1e-7),
            CoupledOptimizerTrainerConfig(
                batch_size=10, encoder_optim_decay=1e-7, decoder_optim_decay=1e-7
            ),
        ]
    )
    def training_config_batch_size(self, request, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path  # this avoids creating a permanent folder
        return request.param

    def test_build_train_data_loader(
        self, model_sample, train_dataset, training_config_batch_size
    ):
        trainer = CoupledOptimizerTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config_batch_size,
        )

        train_data_loader = trainer.get_train_dataloader(train_dataset)

        assert issubclass(type(train_data_loader), torch.utils.data.DataLoader)
        assert train_data_loader.dataset == train_dataset

        assert train_data_loader.batch_size == trainer.training_config.batch_size

    def test_build_eval_data_loader(
        self, model_sample, train_dataset, training_config_batch_size
    ):
        trainer = CoupledOptimizerTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config_batch_size,
        )

        train_data_loader = trainer.get_eval_dataloader(train_dataset)

        assert issubclass(type(train_data_loader), torch.utils.data.DataLoader)
        assert train_data_loader.dataset == train_dataset

        assert train_data_loader.batch_size == trainer.training_config.batch_size


class Test_Set_Training_config:
    @pytest.fixture(
        params=[
            CoupledOptimizerTrainerConfig(decoder_optim_decay=0),
            CoupledOptimizerTrainerConfig(
                batch_size=10, learning_rate=1e-5, encoder_optim_decay=0
            ),
        ]
    )
    def training_configs(self, request, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    def test_set_training_config(self, model_sample, train_dataset, training_configs):
        trainer = CoupledOptimizerTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs,
        )

        # check if default config is set
        if training_configs is None:
            assert trainer.training_config == CoupledOptimizerTrainerConfig(
                output_dir="dummy_output_dir"
            )

        else:
            assert trainer.training_config == training_configs


class Test_Build_Optimizer:
    @pytest.fixture(
        params=[
            CoupledOptimizerTrainerConfig(learning_rate=1e-6),
            CoupledOptimizerTrainerConfig(),
        ]
    )
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
        return request.param

    @pytest.fixture(params=[Adagrad, Adam, Adadelta, SGD, RMSprop])
    def optimizers(self, request, model_sample, training_configs_learning_rate):

        encoder_optimizer = request.param(
            model_sample.encoder.parameters(),
            lr=training_configs_learning_rate.learning_rate,
        )
        decoder_optimizer = request.param(
            model_sample.decoder.parameters(),
            lr=training_configs_learning_rate.learning_rate,
        )
        return (encoder_optimizer, decoder_optimizer)

    def test_default_optimizer_building(
        self, model_sample, train_dataset, training_configs_learning_rate
    ):

        trainer = CoupledOptimizerTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
            encoder_optimizer=None,
            decoder_optimizer=None,
        )

        assert issubclass(type(trainer.encoder_optimizer), torch.optim.Adam)
        assert (
            trainer.encoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )

        assert issubclass(type(trainer.decoder_optimizer), torch.optim.Adam)
        assert (
            trainer.decoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )

    def test_set_custom_optimizer(
        self, model_sample, train_dataset, training_configs_learning_rate, optimizers
    ):
        trainer = CoupledOptimizerTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
            encoder_optimizer=optimizers[0],
            decoder_optimizer=optimizers[1],
        )

        assert issubclass(type(trainer.encoder_optimizer), type(optimizers[0]))
        assert (
            trainer.encoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )

        assert issubclass(type(trainer.decoder_optimizer), type(optimizers[1]))
        assert (
            trainer.decoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )


@pytest.mark.slow
class Test_Main_Training:
    @pytest.fixture(params=[CoupledOptimizerTrainerConfig(num_epochs=3)])
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture(
        params=[
            AEConfig(input_dim=(1, 28, 28)),
            RHVAEConfig(input_dim=(1, 28, 28), latent_dim=5),
        ]
    )
    def ae_config(self, request):
        return request.param

    @pytest.fixture
    def custom_encoder(self, ae_config):
        return Encoder_MLP_Custom(ae_config)

    @pytest.fixture
    def custom_decoder(self, ae_config):
        return Decoder_MLP_Custom(ae_config)

    @pytest.fixture(
        params=[
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
        ]
    )
    def ae(self, ae_config, custom_encoder, custom_decoder, request):
        # randomized

        alpha = request.param

        if alpha < 0.25:
            model = AE(ae_config)

        elif 0.25 <= alpha < 0.5:
            model = AE(ae_config, encoder=custom_encoder)

        elif 0.5 <= alpha < 0.75:
            model = AE(ae_config, decoder=custom_decoder)

        else:
            model = AE(ae_config, encoder=custom_encoder, decoder=custom_decoder)

        return model

    @pytest.fixture(params=[None, Adagrad, Adam, Adadelta, SGD, RMSprop])
    def optimizers(self, request, ae, training_configs):
        if request.param is not None:
            encoder_optimizer = request.param(
                ae.encoder.parameters(), lr=training_configs.learning_rate
            )

            decoder_optimizer = request.param(
                ae.decoder.parameters(), lr=training_configs.learning_rate
            )

        else:
            encoder_optimizer = None
            decoder_optimizer = None

        return (encoder_optimizer, decoder_optimizer)

    def test_train_step(self, ae, train_dataset, training_configs, optimizers):
        trainer = CoupledOptimizerTrainer(
            model=ae,
            train_dataset=train_dataset,
            training_config=training_configs,
            encoder_optimizer=optimizers[0],
            decoder_optimizer=optimizers[1],
        )

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        step_1_loss = trainer.train_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    def test_eval_step(self, ae, train_dataset, training_configs, optimizers):
        trainer = CoupledOptimizerTrainer(
            model=ae,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
            encoder_optimizer=optimizers[0],
            decoder_optimizer=optimizers[1],
        )

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        step_1_loss = trainer.eval_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    def test_main_train_loop(
        self, tmpdir, ae, train_dataset, training_configs, optimizers
    ):

        trainer = CoupledOptimizerTrainer(
            model=ae,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
            encoder_optimizer=optimizers[0],
            decoder_optimizer=optimizers[1],
        )

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        trainer.train()

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )


class Test_Logging:
    @pytest.fixture
    def training_config(self, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        return CoupledOptimizerTrainerConfig(output_dir=dir_path, num_epochs=2)

    @pytest.fixture
    def model_sample(self):
        return RHVAE(RHVAEConfig(input_dim=(1, 28, 28)))

    def test_create_log_file(
        self, tmpdir, model_sample, train_dataset, training_config
    ):
        dir_log_path = os.path.join(tmpdir, "dummy_folder")

        trainer = CoupledOptimizerTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config,
        )

        trainer.train(log_output_dir=dir_log_path)

        assert os.path.isdir(dir_log_path)
        assert f"training_logs_{trainer._training_signature}.log" in os.listdir(
            dir_log_path
        )
