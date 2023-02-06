import os
from copy import deepcopy

import pytest
import torch

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
            AdversarialTrainerConfig(),
            AdversarialTrainerConfig(
                per_device_train_batch_size=100,
                per_device_eval_batch_size=35,
            ),
            AdversarialTrainerConfig(
                per_device_train_batch_size=10, per_device_eval_batch_size=3
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

        assert (
            train_data_loader.batch_size
            == trainer.training_config.per_device_train_batch_size
        )

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

        assert (
            eval_data_loader.batch_size
            == trainer.training_config.per_device_eval_batch_size
        )


class Test_Set_Training_config:
    @pytest.fixture(
        params=[
            AdversarialTrainerConfig(),
            AdversarialTrainerConfig(
                per_device_train_batch_size=10,
                per_device_eval_batch_size=10,
                autoencoder_learning_rate=1e-3,
                discriminator_learning_rate=1e-5,
                autoencoder_optimizer_cls="AdamW",
                autoencoder_optimizer_params={"weight_decay": 0.01},
                discriminator_optimizer_cls="SGD",
                discriminator_optimizer_params={"weight_decay": 0.01},
                autoencoder_scheduler_cls="ExponentialLR",
                autoencoder_scheduler_params={"gamma": 0.321},
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
    def test_wrong_optimizer_cls(self):
        with pytest.raises(AttributeError):
            AdversarialTrainerConfig(autoencoder_optimizer_cls="WrongOptim")

        with pytest.raises(AttributeError):
            AdversarialTrainerConfig(discriminator_optimizer_cls="WrongOptim")

    def test_wrong_optimizer_params(self):
        with pytest.raises(TypeError):
            AdversarialTrainerConfig(
                autoencoder_optimizer_cls="Adam",
                autoencoder_optimizer_params={"wrong_config": 1},
            )

        with pytest.raises(TypeError):
            AdversarialTrainerConfig(
                discriminator_optimizer_cls="Adam",
                discriminator_optimizer_params={"wrong_config": 1},
            )

    @pytest.fixture(
        params=[
            AdversarialTrainerConfig(
                autoencoder_learning_rate=1e-2, discriminator_learning_rate=1e-3
            ),
            AdversarialTrainerConfig(),
        ]
    )
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
        return request.param

    @pytest.fixture(
        params=[
            {
                "autoencoder_optimizer_cls": "Adagrad",
                "autoencoder_optimizer_params": {"lr_decay": 0.1},
                "discriminator_optimizer_cls": "AdamW",
                "discriminator_optimizer_params": {"betas": (0.1234, 0.4321)},
            },
            {
                "autoencoder_optimizer_cls": "SGD",
                "autoencoder_optimizer_params": {"momentum": 0.1},
                "discriminator_optimizer_cls": "SGD",
                "discriminator_optimizer_params": {"momentum": 0.9},
            },
            {
                "autoencoder_optimizer_cls": "SGD",
                "autoencoder_optimizer_params": None,
                "discriminator_optimizer_cls": "SGD",
                "discriminator_optimizer_params": None,
            },
        ]
    )
    def optimizer_config(self, request, training_configs_learning_rate):

        optimizer_config = request.param

        # set optim and params to training config
        training_configs_learning_rate.autoencoder_optimizer_cls = optimizer_config[
            "autoencoder_optimizer_cls"
        ]
        training_configs_learning_rate.autoencoder_optimizer_params = optimizer_config[
            "autoencoder_optimizer_params"
        ]
        training_configs_learning_rate.discriminator_optimizer_cls = optimizer_config[
            "discriminator_optimizer_cls"
        ]
        training_configs_learning_rate.discriminator_optimizer_params = (
            optimizer_config["discriminator_optimizer_params"]
        )

        return optimizer_config

    def test_default_optimizer_building(
        self, model_sample, train_dataset, training_configs_learning_rate
    ):

        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_autoencoder_optimizer()
        trainer.set_discriminator_optimizer()

        assert issubclass(type(trainer.autoencoder_optimizer), torch.optim.Adam)
        assert (
            trainer.autoencoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.autoencoder_learning_rate
        )

        assert issubclass(type(trainer.discriminator_optimizer), torch.optim.Adam)
        assert (
            trainer.discriminator_optimizer.defaults["lr"]
            == training_configs_learning_rate.discriminator_learning_rate
        )

    def test_set_custom_optimizer(
        self,
        model_sample,
        train_dataset,
        training_configs_learning_rate,
        optimizer_config,
    ):
        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_autoencoder_optimizer()
        trainer.set_discriminator_optimizer()

        assert issubclass(
            type(trainer.autoencoder_optimizer),
            getattr(torch.optim, optimizer_config["autoencoder_optimizer_cls"]),
        )
        assert (
            trainer.autoencoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.autoencoder_learning_rate
        )
        if optimizer_config["autoencoder_optimizer_params"] is not None:
            assert all(
                [
                    trainer.autoencoder_optimizer.defaults[key]
                    == optimizer_config["autoencoder_optimizer_params"][key]
                    for key in optimizer_config["autoencoder_optimizer_params"].keys()
                ]
            )

        assert issubclass(
            type(trainer.discriminator_optimizer),
            getattr(torch.optim, optimizer_config["discriminator_optimizer_cls"]),
        )
        assert (
            trainer.discriminator_optimizer.defaults["lr"]
            == training_configs_learning_rate.discriminator_learning_rate
        )
        if optimizer_config["discriminator_optimizer_params"] is not None:
            assert all(
                [
                    trainer.discriminator_optimizer.defaults[key]
                    == optimizer_config["discriminator_optimizer_params"][key]
                    for key in optimizer_config["discriminator_optimizer_params"].keys()
                ]
            )


class Test_Build_Scheduler:
    def test_wrong_scheduler_cls(self):
        with pytest.raises(AttributeError):
            AdversarialTrainerConfig(autoencoder_scheduler_cls="WrongOptim")

        with pytest.raises(AttributeError):
            AdversarialTrainerConfig(discriminator_scheduler_cls="WrongOptim")

    def test_wrong_scheduler_params(self):
        with pytest.raises(TypeError):
            AdversarialTrainerConfig(
                autoencoder_scheduler_cls="ReduceLROnPlateau",
                autoencoder_scheduler_params={"wrong_config": 1},
            )

        with pytest.raises(TypeError):
            AdversarialTrainerConfig(
                discriminator_scheduler_cls="ReduceLROnPlateau",
                discriminator_scheduler_params={"wrong_config": 1},
            )

    @pytest.fixture(
        params=[
            AdversarialTrainerConfig(),
            AdversarialTrainerConfig(learning_rate=1e-5),
        ]
    )
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
        return request.param

    @pytest.fixture(
        params=[
            {
                "autoencoder_scheduler_cls": "StepLR",
                "autoencoder_scheduler_params": {"step_size": 1},
                "discriminator_scheduler_cls": "LinearLR",
                "discriminator_scheduler_params": None,
            },
            {
                "autoencoder_scheduler_cls": None,
                "autoencoder_scheduler_params": None,
                "discriminator_scheduler_cls": "ExponentialLR",
                "discriminator_scheduler_params": {"gamma": 0.1},
            },
            {
                "autoencoder_scheduler_cls": "ReduceLROnPlateau",
                "autoencoder_scheduler_params": {"patience": 12},
                "discriminator_scheduler_cls": None,
                "discriminator_scheduler_params": None,
            },
        ]
    )
    def scheduler_config(self, request, training_configs_learning_rate):

        scheduler_config = request.param

        # set scheduler and params to training config
        training_configs_learning_rate.autoencoder_scheduler_cls = scheduler_config[
            "autoencoder_scheduler_cls"
        ]
        training_configs_learning_rate.autoencoder_scheduler_params = scheduler_config[
            "autoencoder_scheduler_params"
        ]
        training_configs_learning_rate.discriminator_scheduler_cls = scheduler_config[
            "discriminator_scheduler_cls"
        ]
        training_configs_learning_rate.discriminator_scheduler_params = (
            scheduler_config["discriminator_scheduler_params"]
        )

        return request.param

    def test_default_scheduler_building(
        self, model_sample, train_dataset, training_configs_learning_rate
    ):

        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_autoencoder_optimizer()
        trainer.set_autoencoder_scheduler()
        trainer.set_discriminator_optimizer()
        trainer.set_discriminator_scheduler()

        assert trainer.autoencoder_scheduler is None
        assert trainer.discriminator_scheduler is None

    def test_set_custom_scheduler(
        self,
        model_sample,
        train_dataset,
        training_configs_learning_rate,
        scheduler_config,
    ):
        trainer = AdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_autoencoder_optimizer()
        trainer.set_autoencoder_scheduler()
        trainer.set_discriminator_optimizer()
        trainer.set_discriminator_scheduler()

        if scheduler_config["autoencoder_scheduler_cls"] is None:
            assert trainer.autoencoder_scheduler is None
        else:
            assert issubclass(
                type(trainer.autoencoder_scheduler),
                getattr(
                    torch.optim.lr_scheduler,
                    scheduler_config["autoencoder_scheduler_cls"],
                ),
            )
            if scheduler_config["autoencoder_scheduler_params"] is not None:
                assert all(
                    [
                        trainer.autoencoder_scheduler.state_dict()[key]
                        == scheduler_config["autoencoder_scheduler_params"][key]
                        for key in scheduler_config[
                            "autoencoder_scheduler_params"
                        ].keys()
                    ]
                )

        if scheduler_config["discriminator_scheduler_cls"] is None:
            assert trainer.discriminator_scheduler is None

        else:
            assert issubclass(
                type(trainer.discriminator_scheduler),
                getattr(
                    torch.optim.lr_scheduler,
                    scheduler_config["discriminator_scheduler_cls"],
                ),
            )
            if scheduler_config["discriminator_scheduler_params"] is not None:
                assert all(
                    [
                        trainer.discriminator_scheduler.state_dict()[key]
                        == scheduler_config["discriminator_scheduler_params"][key]
                        for key in scheduler_config[
                            "discriminator_scheduler_params"
                        ].keys()
                    ]
                )


class Test_Device_Checks:
    def test_set_environ_variable(self):
        os.environ["LOCAL_RANK"] = "1"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["RANK"] = "3"
        os.environ["MASTER_ADDR"] = "314"
        os.environ["MASTER_PORT"] = "222"

        trainer_config = AdversarialTrainerConfig()

        assert int(trainer_config.local_rank) == 1
        assert int(trainer_config.world_size) == 4
        assert int(trainer_config.rank) == 3
        assert trainer_config.master_addr == "314"
        assert trainer_config.master_port == "222"

        del os.environ["LOCAL_RANK"]
        del os.environ["WORLD_SIZE"]
        del os.environ["RANK"]
        del os.environ["MASTER_ADDR"]
        del os.environ["MASTER_PORT"]


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

    @pytest.fixture(
        params=[
            {
                "autoencoder_optimizer_cls": "Adagrad",
                "autoencoder_optimizer_params": {"lr_decay": 0.1},
                "discriminator_optimizer_cls": "AdamW",
                "discriminator_optimizer_params": {"betas": (0.1234, 0.4321)},
            },
            {
                "autoencoder_optimizer_cls": "SGD",
                "autoencoder_optimizer_params": {"momentum": 0.1},
                "discriminator_optimizer_cls": "SGD",
                "discriminator_optimizer_params": {"momentum": 0.9},
            },
            {
                "autoencoder_optimizer_cls": "SGD",
                "autoencoder_optimizer_params": None,
                "discriminator_optimizer_cls": "SGD",
                "discriminator_optimizer_params": None,
            },
        ]
    )
    def optimizer_config(self, request):

        optimizer_config = request.param

        return optimizer_config

    @pytest.fixture(
        params=[
            {
                "autoencoder_scheduler_cls": "LinearLR",
                "autoencoder_scheduler_params": None,
                "discriminator_scheduler_cls": "LinearLR",
                "discriminator_scheduler_params": None,
            },
            {
                "autoencoder_scheduler_cls": None,
                "autoencoder_scheduler_params": None,
                "discriminator_scheduler_cls": "ExponentialLR",
                "discriminator_scheduler_params": {"gamma": 0.13},
            },
            {
                "autoencoder_scheduler_cls": "ReduceLROnPlateau",
                "autoencoder_scheduler_params": {"patience": 12},
                "discriminator_scheduler_cls": None,
                "discriminator_scheduler_params": None,
            },
        ]
    )
    def scheduler_config(self, request):
        return request.param

    @pytest.fixture
    def trainer(
        self, ae, train_dataset, optimizer_config, scheduler_config, training_configs
    ):

        training_configs.autoencoder_optimizer_cls = optimizer_config[
            "autoencoder_optimizer_cls"
        ]
        training_configs.autoencoder_optimizer_params = optimizer_config[
            "autoencoder_optimizer_params"
        ]
        training_configs.discriminator_optimizer_cls = optimizer_config[
            "discriminator_optimizer_cls"
        ]
        training_configs.discriminator_optimizer_params = optimizer_config[
            "discriminator_optimizer_params"
        ]
        training_configs.autoencoder_scheduler_cls = scheduler_config[
            "autoencoder_scheduler_cls"
        ]
        training_configs.autoencoder_scheduler_params = scheduler_config[
            "autoencoder_scheduler_params"
        ]
        training_configs.discriminator_scheduler_cls = scheduler_config[
            "discriminator_scheduler_cls"
        ]
        training_configs.discriminator_scheduler_params = scheduler_config[
            "discriminator_scheduler_params"
        ]

        trainer = AdversarialTrainer(
            model=ae,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
        )

        trainer.prepare_training()

        return trainer

    def test_train_step(self, trainer):

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        step_1_loss = trainer.train_step(epoch=3)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        for key in start_model_state_dict.keys():
            if "encoder" in key:
                assert not torch.equal(
                    step_1_model_state_dict[key], start_model_state_dict[key]
                )

            if "decoder" in key:
                assert not torch.equal(
                    step_1_model_state_dict[key], start_model_state_dict[key]
                )

            if "discriminator" in key:
                assert not torch.equal(
                    step_1_model_state_dict[key], start_model_state_dict[key]
                )

    def test_eval_step(self, trainer):

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

    def test_main_train_loop(self, trainer):

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        trainer.train()

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        for key in start_model_state_dict.keys():
            if "encoder" in key:
                assert not torch.equal(
                    step_1_model_state_dict[key], start_model_state_dict[key]
                )

            if "decoder" in key:
                assert not torch.equal(
                    step_1_model_state_dict[key], start_model_state_dict[key]
                )

            if "discriminator" in key:
                assert not torch.equal(
                    step_1_model_state_dict[key], start_model_state_dict[key]
                )


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
