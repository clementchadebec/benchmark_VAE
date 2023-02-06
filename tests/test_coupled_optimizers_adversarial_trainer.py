import os
from copy import deepcopy

import pytest
import torch

from pythae.models import VAEGAN, VAEGANConfig
from pythae.trainers import (
    CoupledOptimizerAdversarialTrainer,
    CoupledOptimizerAdversarialTrainerConfig,
)
from tests.data.custom_architectures import *

PATH = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def train_dataset():
    return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))


@pytest.fixture()
def model_sample():
    return VAEGAN(VAEGANConfig(input_dim=(1, 28, 28)))


@pytest.fixture
def training_config(tmpdir):
    tmpdir.mkdir("dummy_folder")
    dir_path = os.path.join(tmpdir, "dummy_folder")
    return CoupledOptimizerAdversarialTrainerConfig(output_dir=dir_path)


class Test_DataLoader:
    @pytest.fixture(
        params=[
            CoupledOptimizerAdversarialTrainerConfig(),
            CoupledOptimizerAdversarialTrainerConfig(
                per_device_train_batch_size=100, per_device_eval_batch_size=35
            ),
            CoupledOptimizerAdversarialTrainerConfig(
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
        trainer = CoupledOptimizerAdversarialTrainer(
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
        trainer = CoupledOptimizerAdversarialTrainer(
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
            CoupledOptimizerAdversarialTrainerConfig(),
            CoupledOptimizerAdversarialTrainerConfig(
                per_device_train_batch_size=10,
                per_device_eval_batch_size=3,
                encoder_learning_rate=1e-5,
                decoder_learning_rate=1e-3,
                discriminator_learning_rate=1e-6,
                encoder_optimizer_cls="AdamW",
                encoder_optimizer_params={"weight_decay": 0.01},
                decoder_optimizer_cls="Adam",
                decoder_optimizer_params=None,
                discriminator_optimizer_cls="SGD",
                discriminator_optimizer_params={"weight_decay": 0.01},
                encoder_scheduler_cls="ExponentialLR",
                encoder_scheduler_params={"gamma": 0.321},
            ),
        ]
    )
    def training_configs(self, request, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    def test_set_training_config(self, model_sample, train_dataset, training_configs):
        trainer = CoupledOptimizerAdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs,
        )

        # check if default config is set
        if training_configs is None:
            assert trainer.training_config == CoupledOptimizerAdversarialTrainerConfig(
                output_dir="dummy_output_dir"
            )

        else:
            assert trainer.training_config == training_configs


class Test_Build_Optimizer:
    def test_wrong_optimizer_cls(self):
        with pytest.raises(AttributeError):
            CoupledOptimizerAdversarialTrainerConfig(encoder_optimizer_cls="WrongOptim")

        with pytest.raises(AttributeError):
            CoupledOptimizerAdversarialTrainerConfig(decoder_optimizer_cls="WrongOptim")

        with pytest.raises(AttributeError):
            CoupledOptimizerAdversarialTrainerConfig(
                discriminator_optimizer_cls="WrongOptim"
            )

    def test_wrong_optimizer_params(self):
        with pytest.raises(TypeError):
            CoupledOptimizerAdversarialTrainerConfig(
                encoder_optimizer_cls="Adam",
                encoder_optimizer_params={"wrong_config": 1},
            )

        with pytest.raises(TypeError):
            CoupledOptimizerAdversarialTrainerConfig(
                decoder_optimizer_cls="Adam",
                decoder_optimizer_params={"wrong_config": 1},
            )

        with pytest.raises(TypeError):
            CoupledOptimizerAdversarialTrainerConfig(
                discriminator_optimizer_cls="Adam",
                discriminator_optimizer_params={"wrong_config": 1},
            )

    @pytest.fixture(
        params=[
            CoupledOptimizerAdversarialTrainerConfig(
                encoder_learning_rate=1e-6,
                decoder_learning_rate=1e-5,
                discriminator_learning_rate=1e-3,
            ),
            CoupledOptimizerAdversarialTrainerConfig(),
        ]
    )
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
        return request.param

    @pytest.fixture(
        params=[
            {
                "encoder_optimizer_cls": "Adagrad",
                "encoder_optimizer_params": {"lr_decay": 0.1},
                "decoder_optimizer_cls": "SGD",
                "decoder_optimizer_params": {"momentum": 0.9},
                "discriminator_optimizer_cls": "AdamW",
                "discriminator_optimizer_params": {"betas": (0.1234, 0.4321)},
            },
            {
                "encoder_optimizer_cls": "SGD",
                "encoder_optimizer_params": {"momentum": 0.1},
                "decoder_optimizer_cls": "SGD",
                "decoder_optimizer_params": None,
                "discriminator_optimizer_cls": "SGD",
                "discriminator_optimizer_params": {"momentum": 0.9},
            },
            {
                "encoder_optimizer_cls": "SGD",
                "encoder_optimizer_params": None,
                "decoder_optimizer_cls": "SGD",
                "decoder_optimizer_params": None,
                "discriminator_optimizer_cls": "SGD",
                "discriminator_optimizer_params": None,
            },
        ]
    )
    def optimizer_config(self, request, training_configs_learning_rate):

        optimizer_config = request.param

        # set optim and params to training config
        training_configs_learning_rate.encoder_optimizer_cls = optimizer_config[
            "encoder_optimizer_cls"
        ]
        training_configs_learning_rate.encoder_optimizer_params = optimizer_config[
            "encoder_optimizer_params"
        ]
        training_configs_learning_rate.decoder_optimizer_cls = optimizer_config[
            "decoder_optimizer_cls"
        ]
        training_configs_learning_rate.decoder_optimizer_params = optimizer_config[
            "decoder_optimizer_params"
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

        trainer = CoupledOptimizerAdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_encoder_optimizer()
        trainer.set_decoder_optimizer()
        trainer.set_discriminator_optimizer()

        assert issubclass(type(trainer.encoder_optimizer), torch.optim.Adam)
        assert (
            trainer.encoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.encoder_learning_rate
        )

        assert issubclass(type(trainer.decoder_optimizer), torch.optim.Adam)
        assert (
            trainer.decoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.decoder_learning_rate
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
        trainer = CoupledOptimizerAdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_encoder_optimizer()
        trainer.set_decoder_optimizer()
        trainer.set_discriminator_optimizer()

        assert issubclass(
            type(trainer.encoder_optimizer),
            getattr(torch.optim, optimizer_config["encoder_optimizer_cls"]),
        )
        assert (
            trainer.encoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.encoder_learning_rate
        )
        if optimizer_config["encoder_optimizer_params"] is not None:
            assert all(
                [
                    trainer.encoder_optimizer.defaults[key]
                    == optimizer_config["encoder_optimizer_params"][key]
                    for key in optimizer_config["encoder_optimizer_params"].keys()
                ]
            )

        assert issubclass(
            type(trainer.decoder_optimizer),
            getattr(torch.optim, optimizer_config["decoder_optimizer_cls"]),
        )
        assert (
            trainer.decoder_optimizer.defaults["lr"]
            == training_configs_learning_rate.decoder_learning_rate
        )
        if optimizer_config["decoder_optimizer_params"] is not None:
            assert all(
                [
                    trainer.decoder_optimizer.defaults[key]
                    == optimizer_config["decoder_optimizer_params"][key]
                    for key in optimizer_config["decoder_optimizer_params"].keys()
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
            CoupledOptimizerAdversarialTrainerConfig(encoder_scheduler_cls="WrongOptim")

        with pytest.raises(AttributeError):
            CoupledOptimizerAdversarialTrainerConfig(decoder_scheduler_cls="WrongOptim")

        with pytest.raises(AttributeError):
            CoupledOptimizerAdversarialTrainerConfig(
                discriminator_scheduler_cls="WrongOptim"
            )

    def test_wrong_scheduler_params(self):
        with pytest.raises(TypeError):
            CoupledOptimizerAdversarialTrainerConfig(
                encoder_scheduler_cls="ReduceLROnPlateau",
                encoder_scheduler_params={"wrong_config": 1},
            )

        with pytest.raises(TypeError):
            CoupledOptimizerAdversarialTrainerConfig(
                decoder_scheduler_cls="ReduceLROnPlateau",
                decoder_scheduler_params={"wrong_config": 1},
            )

        with pytest.raises(TypeError):
            CoupledOptimizerAdversarialTrainerConfig(
                discriminator_scheduler_cls="ReduceLROnPlateau",
                discriminator_scheduler_params={"wrong_config": 1},
            )

    @pytest.fixture(
        params=[
            CoupledOptimizerAdversarialTrainerConfig(),
            CoupledOptimizerAdversarialTrainerConfig(learning_rate=1e-5),
        ]
    )
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
        return request.param

    @pytest.fixture(
        params=[
            {
                "encoder_scheduler_cls": "LinearLR",
                "encoder_scheduler_params": None,
                "decoder_scheduler_cls": "LinearLR",
                "decoder_scheduler_params": None,
                "discriminator_scheduler_cls": "LinearLR",
                "discriminator_scheduler_params": None,
            },
            {
                "encoder_scheduler_cls": "StepLR",
                "encoder_scheduler_params": {"step_size": 0.71},
                "decoder_scheduler_cls": "LinearLR",
                "decoder_scheduler_params": None,
                "discriminator_scheduler_cls": "ExponentialLR",
                "discriminator_scheduler_params": {"gamma": 0.1},
            },
            {
                "encoder_scheduler_cls": None,
                "encoder_scheduler_params": {"patience": 12},
                "decoder_scheduler_cls": None,
                "decoder_scheduler_params": None,
                "discriminator_scheduler_cls": None,
                "discriminator_scheduler_params": None,
            },
        ]
    )
    def scheduler_config(self, request, training_configs_learning_rate):

        scheduler_config = request.param

        # set scheduler and params to training config
        training_configs_learning_rate.encoder_scheduler_cls = scheduler_config[
            "encoder_scheduler_cls"
        ]
        training_configs_learning_rate.encoder_scheduler_params = scheduler_config[
            "encoder_scheduler_params"
        ]
        training_configs_learning_rate.decoder_scheduler_cls = scheduler_config[
            "decoder_scheduler_cls"
        ]
        training_configs_learning_rate.decoder_scheduler_params = scheduler_config[
            "decoder_scheduler_params"
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

        trainer = CoupledOptimizerAdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_encoder_optimizer()
        trainer.set_encoder_scheduler()
        trainer.set_decoder_optimizer()
        trainer.set_decoder_scheduler()
        trainer.set_discriminator_optimizer()
        trainer.set_discriminator_scheduler()

        assert trainer.encoder_scheduler is None
        assert trainer.decoder_scheduler is None
        assert trainer.discriminator_scheduler is None

    def test_set_custom_scheduler(
        self,
        model_sample,
        train_dataset,
        training_configs_learning_rate,
        scheduler_config,
    ):
        trainer = CoupledOptimizerAdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_encoder_optimizer()
        trainer.set_encoder_scheduler()
        trainer.set_decoder_optimizer()
        trainer.set_decoder_scheduler()
        trainer.set_discriminator_optimizer()
        trainer.set_discriminator_scheduler()

        if scheduler_config["encoder_scheduler_cls"] is None:
            assert trainer.encoder_scheduler is None
        else:
            assert issubclass(
                type(trainer.encoder_scheduler),
                getattr(
                    torch.optim.lr_scheduler, scheduler_config["encoder_scheduler_cls"]
                ),
            )
            if scheduler_config["encoder_scheduler_params"] is not None:
                assert all(
                    [
                        trainer.encoder_scheduler.state_dict()[key]
                        == scheduler_config["encoder_scheduler_params"][key]
                        for key in scheduler_config["encoder_scheduler_params"].keys()
                    ]
                )

        if scheduler_config["decoder_scheduler_cls"] is None:
            assert trainer.decoder_scheduler is None
        else:
            assert issubclass(
                type(trainer.decoder_scheduler),
                getattr(
                    torch.optim.lr_scheduler, scheduler_config["decoder_scheduler_cls"]
                ),
            )
            if scheduler_config["decoder_scheduler_params"] is not None:
                assert all(
                    [
                        trainer.decoder_scheduler.state_dict()[key]
                        == scheduler_config["decoder_scheduler_params"][key]
                        for key in scheduler_config["decoder_scheduler_params"].keys()
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


@pytest.mark.slow
class Test_Main_Training:
    @pytest.fixture(params=[CoupledOptimizerAdversarialTrainerConfig(num_epochs=3)])
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture(
        params=[
            VAEGANConfig(input_dim=(1, 28, 28)),
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
        return LayeredDiscriminator_MLP_Custom(ae_config)

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
            model = VAEGAN(ae_config)

        elif 0.25 <= alpha < 0.5:
            model = VAEGAN(ae_config, encoder=custom_encoder)

        elif 0.5 <= alpha < 0.75:
            model = VAEGAN(ae_config, decoder=custom_decoder)

        else:
            model = VAEGAN(ae_config, encoder=custom_encoder, decoder=custom_decoder)

        return model

    @pytest.fixture(
        params=[
            {
                "encoder_optimizer_cls": "Adagrad",
                "encoder_optimizer_params": {"lr_decay": 0.1},
                "decoder_optimizer_cls": "SGD",
                "decoder_optimizer_params": {"momentum": 0.9},
                "discriminator_optimizer_cls": "AdamW",
                "discriminator_optimizer_params": {"betas": (0.1234, 0.4321)},
            },
            {
                "encoder_optimizer_cls": "SGD",
                "encoder_optimizer_params": {"momentum": 0.1},
                "decoder_optimizer_cls": "SGD",
                "decoder_optimizer_params": None,
                "discriminator_optimizer_cls": "SGD",
                "discriminator_optimizer_params": {"momentum": 0.9},
            },
            {
                "encoder_optimizer_cls": "SGD",
                "encoder_optimizer_params": None,
                "decoder_optimizer_cls": "SGD",
                "decoder_optimizer_params": None,
                "discriminator_optimizer_cls": "SGD",
                "discriminator_optimizer_params": None,
            },
        ]
    )
    def optimizer_config(self, request):
        return request.param

    @pytest.fixture(
        params=[
            {
                "encoder_scheduler_cls": "LinearLR",
                "encoder_scheduler_params": None,
                "decoder_scheduler_cls": "LinearLR",
                "decoder_scheduler_params": None,
                "discriminator_scheduler_cls": "LinearLR",
                "discriminator_scheduler_params": None,
            },
            {
                "encoder_scheduler_cls": "StepLR",
                "encoder_scheduler_params": {"step_size": 0.71},
                "decoder_scheduler_cls": "LinearLR",
                "decoder_scheduler_params": None,
                "discriminator_scheduler_cls": "ExponentialLR",
                "discriminator_scheduler_params": {"gamma": 0.1},
            },
            {
                "encoder_scheduler_cls": None,
                "encoder_scheduler_params": {"patience": 12},
                "decoder_scheduler_cls": None,
                "decoder_scheduler_params": None,
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

        training_configs.encoder_optimizer_cls = optimizer_config[
            "encoder_optimizer_cls"
        ]
        training_configs.encoder_optimizer_params = optimizer_config[
            "encoder_optimizer_params"
        ]
        training_configs.decoder_optimizer_cls = optimizer_config[
            "decoder_optimizer_cls"
        ]
        training_configs.decoder_optimizer_params = optimizer_config[
            "decoder_optimizer_params"
        ]
        training_configs.discriminator_optimizer_cls = optimizer_config[
            "discriminator_optimizer_cls"
        ]
        training_configs.discriminator_optimizer_params = optimizer_config[
            "discriminator_optimizer_params"
        ]

        training_configs.encoder_scheduler_cls = scheduler_config[
            "encoder_scheduler_cls"
        ]
        training_configs.encoder_scheduler_params = scheduler_config[
            "encoder_scheduler_params"
        ]
        training_configs.decoder_scheduler_cls = scheduler_config[
            "decoder_scheduler_cls"
        ]
        training_configs.decoder_scheduler_params = scheduler_config[
            "decoder_scheduler_params"
        ]
        training_configs.discriminator_scheduler_cls = scheduler_config[
            "discriminator_scheduler_cls"
        ]
        training_configs.discriminator_scheduler_params = scheduler_config[
            "discriminator_scheduler_params"
        ]

        trainer = CoupledOptimizerAdversarialTrainer(
            model=ae,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
        )

        trainer.prepare_training()

        return trainer

    @pytest.fixture(
        params=[torch.randint(2, (3,)), torch.randint(2, (3,)), torch.randint(2, (3,))]
    )
    def optimizer_updates(self, request):
        return request.param

    def test_train_step(self, trainer):

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

    def test_train_2_steps_updates(self, ae, train_dataset, trainer, optimizer_updates):

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        model_output = ae({"data": train_dataset.data.to(device)})

        model_output.update_encoder = False
        model_output.update_decoder = False
        model_output.update_discriminator = False

        trainer._optimizers_step(model_output)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

        model_output = ae({"data": train_dataset.data.to(device)})

        model_output.update_encoder = bool(optimizer_updates[0])
        model_output.update_decoder = bool(optimizer_updates[1])
        model_output.update_discriminator = bool(optimizer_updates[2])

        trainer._optimizers_step(model_output)

        step_2_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        for key in start_model_state_dict.keys():
            if "encoder" in key:
                if bool(optimizer_updates[0]):
                    assert not torch.equal(
                        step_1_model_state_dict[key], step_2_model_state_dict[key]
                    )

                else:
                    assert torch.equal(
                        step_1_model_state_dict[key], step_2_model_state_dict[key]
                    )

            if "decoder" in key:
                if bool(optimizer_updates[1]):
                    assert not torch.equal(
                        step_1_model_state_dict[key], step_2_model_state_dict[key]
                    )

                else:
                    assert torch.equal(
                        step_1_model_state_dict[key], step_2_model_state_dict[key]
                    )

            if "discriminator" in key:
                if bool(optimizer_updates[2]):
                    assert not torch.equal(
                        step_1_model_state_dict[key], step_2_model_state_dict[key]
                    )

                else:
                    assert torch.equal(
                        step_1_model_state_dict[key], step_2_model_state_dict[key]
                    )

    def test_train_step_various_updates(
        self, ae, train_dataset, trainer, optimizer_updates
    ):

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        model_output = ae({"data": train_dataset.data.to(device)})

        model_output.update_encoder = bool(optimizer_updates[0])
        model_output.update_decoder = bool(optimizer_updates[1])
        model_output.update_discriminator = bool(optimizer_updates[2])

        trainer._optimizers_step(model_output)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        for key in start_model_state_dict.keys():
            if "encoder" in key:
                if bool(optimizer_updates[0]):
                    assert not torch.equal(
                        step_1_model_state_dict[key], start_model_state_dict[key]
                    )

                else:
                    assert torch.equal(
                        step_1_model_state_dict[key], start_model_state_dict[key]
                    )

            if "decoder" in key:
                if bool(optimizer_updates[1]):
                    assert not torch.equal(
                        step_1_model_state_dict[key], start_model_state_dict[key]
                    )

                else:
                    assert torch.equal(
                        step_1_model_state_dict[key], start_model_state_dict[key]
                    )

            if "discriminator" in key:
                if bool(optimizer_updates[2]):
                    assert not torch.equal(
                        step_1_model_state_dict[key], start_model_state_dict[key]
                    )

                else:
                    assert torch.equal(
                        step_1_model_state_dict[key], start_model_state_dict[key]
                    )

    def test_eval_step(self, trainer):

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

    def test_main_train_loop(self, trainer):

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
        return CoupledOptimizerAdversarialTrainerConfig(
            output_dir=dir_path, num_epochs=2
        )

    @pytest.fixture
    def model_sample(self):
        return VAEGAN(VAEGANConfig(input_dim=(1, 28, 28)))

    def test_create_log_file(
        self, tmpdir, model_sample, train_dataset, training_config
    ):
        dir_log_path = os.path.join(tmpdir, "dummy_folder")

        trainer = CoupledOptimizerAdversarialTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config,
        )

        trainer.train(log_output_dir=dir_log_path)

        assert os.path.isdir(dir_log_path)
        assert f"training_logs_{trainer._training_signature}.log" in os.listdir(
            dir_log_path
        )
