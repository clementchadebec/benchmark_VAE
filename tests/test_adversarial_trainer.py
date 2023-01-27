import os
from copy import deepcopy

import pytest
import itertools
import torch
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR, LinearLR, ExponentialLR

from pythae.models import Adversarial_AE, Adversarial_AE_Config
from pythae.trainers import AdversarialTrainer, AdversarialTrainerConfig
from tests.data.custom_architectures import *

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def train_dataset():
    return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))


@pytest.fixture()
def model_sample():
    return Adversarial_AE(Adversarial_AE_Config(input_dim=(1, 28, 28)))


@pytest.fixture
def training_config(tmpdir):
    tmpdir.mkdir("dummy_folder")
    dir_path = os.path.join(tmpdir, "dummy_folder")
    return AdversarialTrainerConfig(output_dir=dir_path)


class Test_DataLoader:
    @pytest.fixture(
        params=[
            AdversarialTrainerConfig(autoencoder_optim_decay=0),
            AdversarialTrainerConfig(
                per_device_train_batch_size=100,
                per_device_eval_batch_size=35,
                autoencoder_optim_decay=1e-7),
            AdversarialTrainerConfig(
                per_device_train_batch_size=10,
                per_device_eval_batch_size=3,
                autoencoder_optim_decay=1e-7,
                discriminator_optim_decay=1e-7,
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
        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config_batch_size,
        )

        train_data_loader = trainer.get_train_dataloader(train_dataset)

        assert issubclass(type(train_data_loader), torch.utils.data.DataLoader)
        assert train_data_loader.dataset == train_dataset

        assert train_data_loader.batch_size == trainer.training_config.per_device_train_batch_size

    def test_build_eval_data_loader(
        self, model_sample, train_dataset, training_config_batch_size
    ):
        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config_batch_size,
        )

        eval_data_loader = trainer.get_eval_dataloader(train_dataset)

        assert issubclass(type(eval_data_loader), torch.utils.data.DataLoader)
        assert eval_data_loader.dataset == train_dataset

        assert eval_data_loader.batch_size == trainer.training_config.per_device_eval_batch_size


class Test_Set_Training_config:
    @pytest.fixture(
        params=[
            AdversarialTrainerConfig(autoencoder_optim_decay=0),
            AdversarialTrainerConfig(
                per_device_train_batch_size=10,
                per_device_eval_batch_size=10,
                learning_rate=1e-3,
                autoencoder_optim_decay=0
            ),
        ]
    )
    def training_configs(self, request, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    def test_set_training_config(self, model_sample, train_dataset, training_configs):
        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs,
        )

        # check if default config is set
        if training_configs is None:
            assert trainer.training_config == AdversarialTrainerConfig(
                output_dir="dummy_output_dir"
            )

        else:
            assert trainer.training_config == training_configs


class Test_Build_Optimizer:
    @pytest.fixture(
        params=[
            AdversarialTrainerConfig(learning_rate=1e-2),
            AdversarialTrainerConfig(),
        ]
    )
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
        return request.param

    @pytest.fixture(params=[Adagrad, Adam, Adadelta, SGD, RMSprop])
    def optimizers(self, request, model_sample, training_configs_learning_rate):

        autoencoder_optimizer = request.param(
            model_sample.encoder.parameters(),
            lr=training_configs_learning_rate.learning_rate,
        )
        discriminator_optimizer = request.param(
            model_sample.decoder.parameters(),
            lr=training_configs_learning_rate.learning_rate,
        )
        return (autoencoder_optimizer, discriminator_optimizer)

    def test_default_optimizer_building(
        self, model_sample, train_dataset, training_configs_learning_rate
    ):

        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
            autoencoder_optimizer=None,
            discriminator_optimizer=None,
        )

        assert issubclass(type(trainer.autoencoder_optimizer), torch.optim.Adam)
        assert (
            trainer.autoencoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )

        assert issubclass(type(trainer.discriminator_optimizer), torch.optim.Adam)
        assert (
            trainer.discriminator_optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )

    def test_set_custom_optimizer(
        self, model_sample, train_dataset, training_configs_learning_rate, optimizers
    ):
        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
            autoencoder_optimizer=optimizers[0],
            discriminator_optimizer=optimizers[1],
        )

        assert issubclass(type(trainer.autoencoder_optimizer), type(optimizers[0]))
        assert (
            trainer.autoencoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )

        assert issubclass(type(trainer.discriminator_optimizer), type(optimizers[1]))
        assert (
            trainer.discriminator_optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )

class Test_Build_Scheduler:
    @pytest.fixture(params=[AdversarialTrainerConfig(), AdversarialTrainerConfig(learning_rate=1e-5)])
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
        return request.param

    @pytest.fixture(params=[Adagrad, Adam, Adadelta, SGD, RMSprop])
    def optimizers(self, request, model_sample, training_configs_learning_rate):

        autoencoder_optimizer = request.param(
            model_sample.encoder.parameters(),
            lr=training_configs_learning_rate.learning_rate,
        )
        discriminator_optimizer = request.param(
            model_sample.decoder.parameters(),
            lr=training_configs_learning_rate.learning_rate,
        )
        return (autoencoder_optimizer, discriminator_optimizer)

    @pytest.fixture(
        params=[
            (StepLR, {"step_size": 1}),
            (LinearLR, {"start_factor": 0.01}),
            (ExponentialLR, {"gamma": 0.1}),
        ]
    )
    def schedulers(
        self, request, optimizers
    ):
        if request.param[0] is not None:
            autoencoder_scheduler = request.param[0](optimizers[0], **request.param[1])
            discriminator_scheduler = request.param[0](optimizers[1], **request.param[1])

        else:
            autoencoder_scheduler = None
            discriminator_scheduler = None

        return (autoencoder_scheduler, discriminator_scheduler)

    def test_default_scheduler_building(
        self, model_sample, train_dataset, training_configs_learning_rate
    ):

        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
            autoencoder_optimizer=None,
            discriminator_optimizer=None
        )

        assert issubclass(
            type(trainer.autoencoder_scheduler), torch.optim.lr_scheduler.ReduceLROnPlateau
        )

        assert issubclass(
            type(trainer.discriminator_scheduler), torch.optim.lr_scheduler.ReduceLROnPlateau
        )

    def test_set_custom_scheduler(
        self,
        model_sample,
        train_dataset,
        training_configs_learning_rate,
        optimizers,
        schedulers,
    ):
        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
            autoencoder_optimizer=optimizers[0],
            autoencoder_scheduler=schedulers[0],
            discriminator_optimizer=optimizers[1],
            discriminator_scheduler=schedulers[1]

        )

        assert issubclass(type(trainer.autoencoder_scheduler), type(schedulers[0]))
        assert issubclass(type(trainer.discriminator_scheduler), type(schedulers[1]))


@pytest.mark.slow
class Test_Main_Training:
    @pytest.fixture(params=[AdversarialTrainerConfig(num_epochs=3, learning_rate=1e-3)])
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture(
        params=[
            Adversarial_AE_Config(input_dim=(1, 28, 28), adversarial_loss_scale=0.2),
            Adversarial_AE_Config(input_dim=(1, 28, 28), latent_dim=5),
        ]
    )
    def ae_config(self, request):
        return request.param

    @pytest.fixture
    def custom_encoder(self, ae_config):
        return Encoder_VAE_MLP_Custom(ae_config)

    @pytest.fixture
    def custom_decoder(self, ae_config):
        return Decoder_MLP_Custom(ae_config)

    @pytest.fixture
    def custom_discriminator(self, ae_config):
        ae_config.discriminator_input_dim = ae_config.latent_dim
        return Discriminator_MLP_Custom(ae_config)

    @pytest.fixture(
        params=[
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
        ]
    )
    def ae(
        self,
        ae_config,
        custom_encoder,
        custom_decoder,
        custom_discriminator,
        request,
    ):
        # randomized

        alpha = request.param

        if alpha < 0.125:
            model = Adversarial_AE(ae_config)

        elif 0.125 <= alpha < 0.25:
            model = Adversarial_AE(ae_config, encoder=custom_encoder)

        elif 0.25 <= alpha < 0.375:
            model = Adversarial_AE(ae_config, decoder=custom_decoder)

        elif 0.375 <= alpha < 0.5:
            model = Adversarial_AE(ae_config, discriminator=custom_discriminator)

        elif 0.5 <= alpha < 0.625:
            model = Adversarial_AE(
                ae_config, encoder=custom_encoder, decoder=custom_decoder
            )

        elif 0.625 <= alpha < 0:
            model = Adversarial_AE(
                ae_config, encoder=custom_encoder, discriminator=custom_discriminator
            )

        elif 0.750 <= alpha < 0.875:
            model = Adversarial_AE(
                ae_config, decoder=custom_decoder, discriminator=custom_discriminator
            )

        else:
            model = Adversarial_AE(
                ae_config,
                encoder=custom_encoder,
                decoder=custom_decoder,
                discriminator=custom_discriminator,
            )

        return model

    @pytest.fixture(params=[None, Adagrad, Adam, Adadelta, SGD, RMSprop])
    def optimizers(self, request, ae, training_configs):
        if request.param is not None:
            autoencoder_optimizer = request.param(
                itertools.chain(ae.encoder.parameters(), ae.decoder.parameters()), lr=training_configs.learning_rate
            )

            discriminator_optimizer = request.param(
                ae.discriminator.parameters(), lr=training_configs.learning_rate
            )

        else:
            autoencoder_optimizer = None
            discriminator_optimizer = None

        return (autoencoder_optimizer, discriminator_optimizer)

    @pytest.fixture(
        params=[
            (None, None),
            (StepLR, {"step_size": 1, "gamma": 0.99}),
            (LinearLR, {"start_factor": 0.99}),
            (ExponentialLR, {"gamma": 0.99}),
        ]
    )
    def schedulers(self, request, optimizers):
        if request.param[0] is not None and optimizers[0] is not None:
            autoencoder_scheduler = request.param[0](optimizers[0], **request.param[1])
        
        else:
            autoencoder_scheduler = None
        
        if request.param[0] is not None and optimizers[1] is not None:
            discriminator_scheduler = request.param[0](optimizers[1], **request.param[1])

        else:
            discriminator_scheduler = None

        return (autoencoder_scheduler, discriminator_scheduler)


    def test_train_step(self, ae, train_dataset, training_configs, optimizers, schedulers):
        trainer = AdversarialTrainer(
            model=ae,
            train_dataset=train_dataset,
            training_config=training_configs,
            autoencoder_optimizer=optimizers[0],
            discriminator_optimizer=optimizers[1],
            autoencoder_scheduler=schedulers[0],
            discriminator_scheduler=schedulers[1]
        )

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        step_1_loss = trainer.train_step(epoch=3)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        for key in start_model_state_dict.keys():
            if "encoder" in key:
                assert not torch.equal(step_1_model_state_dict[key], start_model_state_dict[key])

            if "decoder" in key:
                assert not torch.equal(step_1_model_state_dict[key], start_model_state_dict[key])

            if "discriminator" in key:
                assert not torch.equal(step_1_model_state_dict[key], start_model_state_dict[key])


    def test_eval_step(self, ae, train_dataset, training_configs, optimizers, schedulers):
        trainer = AdversarialTrainer(
            model=ae,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
            autoencoder_optimizer=optimizers[0],
            discriminator_optimizer=optimizers[1],
            autoencoder_scheduler=schedulers[0],
            discriminator_scheduler=schedulers[1]
        )

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        step_1_loss = trainer.eval_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights not were updated
        assert all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    def test_main_train_loop(
        self, ae, train_dataset, training_configs, optimizers, schedulers
    ):

        trainer = AdversarialTrainer(
            model=ae,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
            autoencoder_optimizer=optimizers[0],
            discriminator_optimizer=optimizers[1],
            autoencoder_scheduler=schedulers[0],
            discriminator_scheduler=schedulers[1]
        )

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        trainer.train()

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        for key in start_model_state_dict.keys():
            if "encoder" in key:
                assert not torch.equal(step_1_model_state_dict[key], start_model_state_dict[key])

            if "decoder" in key:
                assert not torch.equal(step_1_model_state_dict[key], start_model_state_dict[key])

            if "discriminator" in key:
                assert not torch.equal(step_1_model_state_dict[key], start_model_state_dict[key])

class Test_Logging:
    @pytest.fixture
    def training_config(self, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        return AdversarialTrainerConfig(output_dir=dir_path, num_epochs=2)

    @pytest.fixture
    def model_sample(self):
        return Adversarial_AE(Adversarial_AE_Config(input_dim=(1, 28, 28)))

    def test_create_log_file(
        self, tmpdir, model_sample, train_dataset, training_config
    ):
        dir_log_path = os.path.join(tmpdir, "dummy_folder")

        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config,
        )

        trainer.train(log_output_dir=dir_log_path)

        assert os.path.isdir(dir_log_path)
        assert f"training_logs_{trainer._training_signature}.log" in os.listdir(
            dir_log_path
        )
